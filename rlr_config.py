import os
import sys
import warnings
from trl.core import flatten_dict
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple
from transformers import is_bitsandbytes_available, is_torchvision_available


@dataclass
class RLR_Config:
    r"""
    Configuration class for the [`RLRTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        exp_name (`str`, *optional*, defaults to `os.path.basename(sys.argv[0])[: -len(".py")]`):
            Name of this experiment (defaults to the file name without the extension).
        run_name (`str`, *optional*, defaults to `""`):
            Name of this run.
        log_with (`Optional[Literal["wandb", "tensorboard"]]`, *optional*, defaults to `None`):
            Log with either `"wandb"` or `"tensorboard"`. Check
            [tracking](https://huggingface.co/docs/accelerate/usage_guides/tracking) for more details.
        log_image_freq (`int`, *optional*, defaults to `1`):
            Frequency for logging images.
        tracker_kwargs (`Dict[str, Any]`, *optional*, defaults to `{}`):
            Keyword arguments for the tracker (e.g., `wandb_project`).
        accelerator_kwargs (`Dict[str, Any]`, *optional*, defaults to `{}`):
            Keyword arguments for the accelerator.
        project_kwargs (`Dict[str, Any]`, *optional*, defaults to `{}`):
            Keyword arguments for the accelerator project config (e.g., `logging_dir`).
        tracker_project_name (`str`, *optional*, defaults to `"trl"`):
            Name of project to use for tracking.
        logdir (`str`, *optional*, defaults to `"logs"`):
            Top-level logging directory for checkpoint saving.
        num_epochs (`int`, *optional*, defaults to `100`):
            Number of epochs to train.
        save_freq (`int`, *optional*, defaults to `1`):
            Number of epochs between saving model checkpoints.
        num_checkpoint_limit (`int`, *optional*, defaults to `5`):
            Number of checkpoints to keep before overwriting old ones.
        mixed_precision (`str`, *optional*, defaults to `"fp16"`):
            Mixed precision training.
        allow_tf32 (`bool`, *optional*, defaults to `True`):
            Allow `tf32` on Ampere GPUs.
        resume_from (`str`, *optional*, defaults to `""`):
            Path to resume training from a checkpoint.
        sample_num_steps (`int`, *optional*, defaults to `50`):
            Number of sampler inference steps.
        sample_eta (`float`, *optional*, defaults to `1.0`):
            Eta parameter for the DDIM sampler.
        sample_guidance_scale (`float`, *optional*, defaults to `5.0`):
            Classifier-free guidance weight.
        train_use_8bit_adam (`bool`, *optional*, defaults to `False`):
            Whether to use the 8bit Adam optimizer from `bitsandbytes`.
        train_learning_rate (`float`, *optional*, defaults to `1e-3`):
            Learning rate.
        train_adam_beta1 (`float`, *optional*, defaults to `0.9`):
            Beta1 for Adam optimizer.
        train_adam_beta2 (`float`, *optional*, defaults to `0.999`):
            Beta2 for Adam optimizer.
        train_adam_weight_decay (`float`, *optional*, defaults to `1e-4`):
            Weight decay for Adam optimizer.
        train_adam_epsilon (`float`, *optional*, defaults to `1e-8`):
            Epsilon value for Adam optimizer.
        train_gradient_accumulation_steps (`int`, *optional*, defaults to `1`):
            Number of gradient accumulation steps.
        train_max_grad_norm (`float`, *optional*, defaults to `1.0`):
            Maximum gradient norm for gradient clipping.
        negative_prompts (`Optional[str]`, *optional*, defaults to `None`):
            Comma-separated list of prompts to use as negative examples.
        truncated_backprop_rand (`bool`, *optional*, defaults to `True`):
            If `True`, randomized truncation to different diffusion timesteps is used.
        truncated_backprop_timestep (`int`, *optional*, defaults to `49`):
            Absolute timestep to which the gradients are backpropagated. Used only if `truncated_backprop_rand=False`.
        truncated_rand_backprop_minmax (`Tuple[int, int]`, *optional*, defaults to `(0, 50)`):
            Range of diffusion timesteps for randomized truncated backpropagation.
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the final model to the Hub.
    """

    exp_name: str = os.path.basename(sys.argv[0])[: -len(".py")]
    run_name: str = ""
    seed: int = 0
    log_with: Optional[Literal["wandb", "tensorboard"]] = None
    log_image_freq: int = 1
    tracker_kwargs: Dict[str, Any] = field(default_factory=dict)
    accelerator_kwargs: Dict[str, Any] = field(default_factory=dict)
    project_kwargs: Dict[str, Any] = field(default_factory=dict)
    tracker_project_name: str = "trl"
    logdir: str = "logs"
    num_epochs: int = 100
    save_freq: int = 4
    num_checkpoint_limit: int = 5
    mixed_precision: str = "fp16"
    allow_tf32: bool = True
    resume_from: str = ""
    sample_num_steps: int = 50
    zo_eps: float = 1e-3
    sample_num_batches_per_epoch: int = 1
    sample_batch_size: int = 16
    reward_fn: str = 'hps'  # Can be "hps", "aesthetic", "hps_aesthetic", "pickscore", or "imagereward"
    async_reward_computation: bool = False
    grad_scale: float = 1
    loss_coeff: float = 0.05
    ho_loss_coeff: float = 1.0 # 直接把这个系数设成了0，似乎会导致最后生成的东西变得一样了
    zo_loss_coeff: float = 0.001
    aesthetic_target: float = 10
    sample_eta: float = 1.0
    sample_guidance_scale: float = 5.0
    chain_len: int = 1 # chain for one RL steps
    prompt_fn: str = 'simple_animals'
    gradient_estimation_strategy: str = 'gaussian'    # gaussian, uniform, fixed, RLR, RL, LR, PG
    pure_ZO: bool = False # if True, only use ZO for gradient estimation, otherwise is the RLR estimation
    backprop_kwargs = {'gaussian': {'mean': 42, 'std': 5}, 'uniform': {'min': 0, 'max': 50}, 'fixed': {'value': 48}, 'RLR':{'value': 49}, 'RL':{None}, 'LR':{None}, 'PG':{None}}
    
    train_batch_size: int = 2
    eval_batch_size: int = 8
    train_use_8bit_adam: bool = False
    train_learning_rate: float = 1e-3 # 1e-3 for supervised learning; 3e-4 for RL; 1e-6 for ZO
    train_adam_beta1: float = 0.9
    train_adam_beta2: float = 0.999
    train_adam_weight_decay: float = 1e-4
    train_adam_epsilon: float = 1e-8
    train_gradient_accumulation_steps: int = 1
    train_zo_sample_budget: int = 1
    train_max_grad_norm: float = 1.0
    negative_prompts: Optional[str] = None
    train_clip_range: float = 1e-4
    train_adv_clip_max: float = 5
    per_prompt_stat_tracking: bool = False
    train_num_inner_epochs: int = 1
    train_cfg: bool = True
    timestep_fraction: float = 1.0
    
    push_to_hub: bool = False

    def to_dict(self):
        output_dict = {}
        for key, value in self.__dict__.items():
            output_dict[key] = value
        return flatten_dict(output_dict)

    def __post_init__(self):
        if self.log_with not in ["wandb", "tensorboard"]:
            warnings.warn(
                "Accelerator tracking only supports image logging if `log_with` is set to 'wandb' or 'tensorboard'."
            )

        if self.log_with == "wandb" and not is_torchvision_available():
            warnings.warn("Wandb image logging requires torchvision to be installed")

        if self.train_use_8bit_adam and not is_bitsandbytes_available():
            raise ImportError(
                "You need to install bitsandbytes to use 8bit Adam. "
                "You can install it with `pip install bitsandbytes`."
            )
