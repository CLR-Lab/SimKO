<div align="center">

# SimKO: Simple Pass@K Policy Optimization

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)]()

</div>

## News
- **[2025/10/17]** We release our [paper]() and code. 🚀
## Quick Start
### Installation


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
SimKO: specify `topk`, `mix_topk_coef`, `tau` and `simko` in `run_qwen2.5-math-7b_SimKO.sh` to train the model with SimKO.
```
bash run_qwen2.5-math-7b_SimKO.sh
```

GRPO
```
bash run_qwen2.5-math-7b_grpo.sh
```

## Acknowledgement
The code is based  on [RLVR-Decomposed](https://github.com/TianHongZXY/RLVR-Decomposed). 


## Citation

If you find our paper or code useful, please consider cite our work:

```bibtex

```
