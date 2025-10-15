<div align="center">

# SimKO: Simple Pass@K Policy Optimization

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2506.01347)

</div>

## News
- **[2025/10/16]** We release our [paper]() and code. 🚀
## Quick Start
### Installation
Our code is implemented based on [verl](https://github.com/volcengine/verl). We recommend to use docker image provided by verl, please refer to their [documents](https://verl.readthedocs.io/en/v0.2.x/start/install.html).

Start from a custom environment:
```
conda create -y -n verl python=3.10.14 && conda activate verl
pip install -e .
pip install vllm==0.8.2
pip install latex2sympy2
pip install fire
pip install tensordict==0.7.2
python -m pip install flash-attn --no-build-isolation
```


## Training
SimKO: specify `topk`, `mix_topk_coef` and `simko` in `run_qwen2.5-math-7b_SimKO.sh` to train the model with SimKO.
```
bash run_qwen2.5-math-7b_psr_nsr.sh
```

GRPO
```
bash run_qwen2.5-math-7b_grpo.sh
```



 ## Citation

If you find our paper or code useful, please consider cite our work:

```bibtex

```
