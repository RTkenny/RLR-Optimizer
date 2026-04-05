# **Recursive Likelihood Ratio Optimizer**
![RLR](figures/rlr_c.png)

Official implementation of the paper:  
**Half-order Fine-Tuning for Diffusion Model: A Recursive Likelihood Ratio Optimizer [arXiv Preprint](https://arxiv.org/abs/2502.00639)**


## Overview
The Recursive Likelihood Ratio (RLR) Optimizer is a novel fine-tuning algorithm for diffusion models that intelligently balances computational efficiency with gradient estimation accuracy. Informed by insights from zeroth-order (ZO) optimization, RLR introduces a specially designed half-order (HO) estimator that achieves **unbiased gradient estimation** with **significantly reduced variance** compared to standard ZO methods. By seamlessly integrating this optimized HO estimator with first-order (FO) and ZO estimators in a recursive framework, RLR enables stable and efficient fine-tuning of diffusion models across all time steps. This approach allows RLR to overcome the high memory cost of full backpropagation and the sample inefficiency of reinforcement learning, ensuring robust alignment of diffusion models to downstream tasks with minimal computational overhead.


## News

Our paper has been accepted as oral presentation in ICLR 2026. The implementation is open-source!

## Model Preparation

For post-training, download the following pretrained models:

- Stable Diffusion v1.4: `CompVis/stable-diffusion-v1-4`
- CLIP ViT-L/14: `openai/clip-vit-large-patch14`

After downloading, update the corresponding model paths in:

- `aesthetic_scorer.py`
- `main.py`

## Environment Setup

The experiments are tested with:

- `trl==0.12.2`
- `transformers==4.46.3`

```bash
cd ./code
conda create -n RLR python=3.10 -y
conda activate RLR

pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu122
pip install -r requirements.txt
```

## Training Commands

Run PPO baseline (RL):

```bash
bash ./scripts/aesthetic_ppo.sh
```

Run Stable Diffusion v1.4 training with the RLR optimizer:

```bash
bash ./scripts/aesthetic_RLR_vb.sh
```

## Notes

To train with truncated backpropagation (AlignProp style), set `chain_len=0` in `scripts/aesthetic_RLR_vb.sh`.

## Citation
If you find this work useful, please cite:

```bibtex
@article{ren2025half,
  title={Half-order Fine-Tuning for Diffusion Model: A Recursive Likelihood Ratio Optimizer},
  author={Ren, Tao and Zhang, Zishi and Jiang, Jingyang and Li, Zehao and Qin, Shentao and Zheng, Yi and Li, Guanghao and Sun, Qianyou and Li, Yan and Liang, Jiafeng and others},
  journal={arXiv preprint arXiv:2502.00639},
  year={2025}
}
```