# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']

def row_quantile_masked(x: torch.Tensor, mask: torch.Tensor, q: float, eps=1e-8):
    B, T = x.shape
    qs = []
    for b in range(B):
        xb = x[b][mask[b]]
        if xb.numel() == 0:
            xb = x[b]
        qs.append(torch.quantile(xb, q))
    return torch.stack(qs, dim=0)  # [B]
class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

    def _forward_micro_batch(self, micro_batch, temperature, simko=False,top_k=1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            max_token: # (bs, response_len) - if simko=True
            top_k_log_probs: # (bs, response_len, top_k) - if simko=True
        """
        response_length = micro_batch['responses'].size(-1)
        multi_modal_inputs = {}
        if 'multi_modal_inputs' in micro_batch:
            for key in micro_batch['multi_modal_inputs'][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch['multi_modal_inputs']],
                                                    dim=0)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."),
                                                          indices).transpose(0, 1).unsqueeze(
                                                              1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                          indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           **multi_modal_inputs,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                predicted_tokens_rmpad = torch.argmax(logits_rmpad, dim=-1)  # (total_nnz,)
                max_token_rmpad = (predicted_tokens_rmpad == input_ids_rmpad_rolled).float()  # (total_nnz,)

                if simko:
                    topk_idx_rmpad = torch.topk(logits_rmpad.detach(), k=top_k, dim=-1).indices   # (total_nnz, K)

    
                    total_nnz, vocab_size = logits_rmpad.shape
                    K = topk_idx_rmpad.size(-1)

                    logits_exp = logits_rmpad.unsqueeze(1).expand(total_nnz, K, vocab_size).reshape(total_nnz*K, vocab_size)  # (N*K, V)
                    labels_exp = topk_idx_rmpad.reshape(total_nnz*K)                                                          # (N*K,)

           
                    topk_logp_flat = logprobs_from_logits(logits_exp, labels_exp)   # (N*K,)

                 
                    topk_logp_rmpad = topk_logp_flat.view(total_nnz, K)             # (total_nnz, K)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                                    # 新增：gather token accuracy
                    max_token_rmpad = gather_outpus_and_unpad(max_token_rmpad,
                                                              gather_dim=0,
                                                              unpad_dim=0,
                                                              padding_size=pad_size)
                    if simko:
                        topk_logp_rmpad = gather_outpus_and_unpad(topk_logp_rmpad,
                                                                gather_dim=0,
                                                                unpad_dim=0,
                                                                padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)
                # 新增：pad token accuracy
                full_max_token = pad_input(hidden_states=max_token_rmpad.unsqueeze(-1),
                                            indices=indices,
                                            batch=batch_size,
                                            seqlen=seqlen)
                if simko:
                    full_topk_logp = pad_input(hidden_states=topk_logp_rmpad,
                                            indices=indices,
                                            batch=batch_size,
                                            seqlen=seqlen)
                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                max_token = full_max_token.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                if simko:
                    topk_log_probs = full_topk_logp[:, -response_length - 1:-1, :]  # (bs, response_len, K)
            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           **multi_modal_inputs,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
              
                predicted_tokens = torch.argmax(logits, dim=-1)  # (bsz, response_length)
                max_token = (predicted_tokens == micro_batch['responses']).float()  # (bsz, response_length)
                if simko and top_k > 0:
                    # logits: (bs, Tresp, V)
                    B, T, V = logits.shape
                    topk_idx = torch.topk(logits.detach(), k=top_k, dim=-1).indices                   # (B, T, K)
                    K = topk_idx.size(-1)

                    logits_2d = logits.reshape(B*T, V)                                                # (N, V)
                    topk_idx_2d = topk_idx.reshape(B*T, K)                                            # (N, K)
                    logits_exp = logits_2d.unsqueeze(1).expand(B*T, K, V).reshape(B*T*K, V)           # (N*K, V)
                    labels_exp = topk_idx_2d.reshape(B*T*K)                                           # (N*K,)

        
                    topk_logp_flat = logprobs_from_logits(logits_exp, labels_exp)                     # (N*K,)

                    topk_log_probs = topk_logp_flat.view(B, T, K)   
            if simko:                                  # (bs, response_len, K)
                return entropy, log_probs, max_token, topk_log_probs
            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto, return_entropy: bool = False,simko: bool = False,top_k:int = 1) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []

        topk_log_probs_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
         
                if simko:
                    entropy, log_probs, max_token,topk_log_probs = self._forward_micro_batch(micro_batch, temperature=temperature,simko=True,top_k=top_k)
                else:
                    entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)

            log_probs_lst.append(log_probs)
            entropy_lst.append(entropy)
       
            if simko:
                topk_log_probs_lst.append(topk_log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)
        entropy = torch.concat(entropy_lst, dim=0)
        if simko:
            topk_log_probs = torch.concat(topk_log_probs_lst, dim=0)
  

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
        
            if simko:
                topk_log_probs = topk_log_probs[revert_indices]
        if simko:
            return log_probs, entropy, topk_log_probs
        if data.meta_info.get('return_entropy', False):
            return log_probs, entropy
        return log_probs

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages', 'token_level_scores']
        if self.config.simko:
            select_keys.append('old_log_probs_topk')
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                # print(mini_batch)
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    # print(len(  data))
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())  # actor device is cpu when using offload
                    responses = data['responses']
                    response_length = responses.size(1)
                    attention_mask = data['attention_mask']
                    response_mask = attention_mask[:, -response_length:]
                    old_log_prob = data['old_log_probs']
                    advantages = data['advantages']
                    token_level_scores = data['token_level_scores']
                    clip_ratio = self.config.clip_ratio
                    entropy_coeff = self.config.entropy_coeff
                    positive_learning_weight = self.config.positive_learning_weight

                    # all return: (bsz, response_length)
                    


                    if self.config.simko:
                        entropy, log_prob, max_token,topk_log_probs = self._forward_micro_batch(micro_batch=data, temperature=temperature,simko=True,top_k=self.config.top_k)
                        entropy = entropy.detach()
                    else:
                        entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)
                        
                    if self.config.simko:
                        old_log_probs_topk=data['old_log_probs_topk']
                        entropy = entropy.detach()
                        pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss_simko(old_log_prob=old_log_prob,
                                                                                    old_log_probs_topk=old_log_probs_topk,
                                                                                    log_prob=log_prob,
                                                                                    topk_log_probs=topk_log_probs,
                                                                                    entropy=entropy,
                                                                                    advantages=advantages,
                                                                                    eos_mask=response_mask,
                                                                                    cliprange=clip_ratio,
                                                                                    token_level_scores=token_level_scores,
                                                                                    max_token=max_token,
                                                                                    mix_topk_coef=self.config.mix_topk_coef,
                                                                                    )
                    else:
                        pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                                    log_prob=log_prob,
                                                                                    advantages=advantages,
                                                                                    eos_mask=response_mask,
                                                                                    cliprange=clip_ratio,
                                                                                    token_level_scores=token_level_scores,
                                                                                    positive_learning_weight=positive_learning_weight)
                    # compute entropy loss from entropy
                    entropy_loss = verl_F.masked_mean(entropy, response_mask)

                    # compute policy loss
                    policy_loss = pg_loss - entropy_loss * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = data['ref_log_prob']
                        # compute kl loss
                        kld = core_algos.kl_penalty(logprob=log_prob,
                                                    ref_logprob=ref_log_prob,
                                                    kl_penalty=self.config.kl_loss_type)
                        kl_loss = masked_mean(kld, response_mask)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics['actor/kl_loss'] = kl_loss.detach().item()
                        metrics['actor/kl_coef'] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
          
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                     
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        'actor/entropy_loss': entropy_loss.detach().item(),
                        'actor/pg_loss': pg_loss.detach().item(),
                        'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                        'actor/ppo_kl': ppo_kl.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
