import ipdb
st = ipdb.set_trace
import builtins
import time
import datetime
import os
builtins.st = ipdb.set_trace
from dataclasses import dataclass, field
import prompts as prompts_file
import numpy as np
from transformers import HfArgumentParser, is_wandb_available

# import debugpy
# debugpy.listen(5678)               # 在本机 5678 端口打开监听
# print("🔍 Debugger is listening on port 5678, waiting to attach...")
# debugpy.wait_for_client()          # 这里会阻塞，直到 VSCode 连接

from rlr_config import RLR_Config
from rlr_trainer_v_b import RLR_Trainer
from sd_pipeline import DiffusionPipeline
from trl.models.auxiliary_modules import aesthetic_scorer
import tempfile
from PIL import Image
if is_wandb_available():
    import wandb

@dataclass
class ScriptArguments:
    pretrained_model: str = field(
        default="/root/autodl-tmp/CompVis/stable-diffusion-v1-4", metadata={"help": "the pretrained model to use"} # "runwayml/stable-diffusion-v1-5" original /root/autodl-tmp/CompVis/stable-diffusion-v1-4 'stabilityai/stable-diffusion-2-base'
    )
    pretrained_revision: str = field(default="main", metadata={"help": "the pretrained model revision to use"})
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})

def image_outputs_logger(image_data, global_step, accelerate_logger):
    images = image_data[0][0]
    prompts = image_data[0][1]

    if hasattr(accelerate_logger, "log_images"):
        result = {}
        for i, image in enumerate(images):
            prompt = str(prompts[i]) if i < len(prompts) else f"image_{i}"
            result[f"{i:02d}_{prompt[:25]}"] = image.unsqueeze(0).float().detach().cpu()
        accelerate_logger.log_images(result, step=global_step)
        return

    tracker = getattr(accelerate_logger, "tracker", None)
    if tracker is not None and hasattr(tracker, "add_image"):
        for i, image in enumerate(images):
            prompt = str(prompts[i]) if i < len(prompts) else f"image_{i}"
            tag = f"images/{i:02d}_{prompt[:25]}".replace(" ", "_")
            tracker.add_image(tag, image.detach().cpu().float().clamp(0, 1), global_step)
        if hasattr(tracker, "flush"):
            tracker.flush()
        return

    if is_wandb_available() and hasattr(accelerate_logger, "log"):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, image in enumerate(images):
                pil = Image.fromarray(
                    (image.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil.save(os.path.join(tmpdir, f"{i}.jpg"))
            accelerate_logger.log(
                {
                    "images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{i}.jpg"),
                            caption=f"{str(prompt)[:25]}",
                        )
                        for i, prompt in enumerate(prompts)
                    ],
                },
                step=global_step,
            )


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, RLR_Config))
    script_args, training_args = parser.parse_args_into_dataclasses()
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    # if not config.run_name:
    #     config.run_name = unique_id
    # else:
    #     config.run_name += "_" + unique_id
    project_dir = f"{unique_id}_{training_args.reward_fn}_{training_args.gradient_estimation_strategy}_seed_{training_args.seed}"
    os.makedirs(f"checkpoints/{project_dir}", exist_ok=True)
    
    training_args.project_kwargs = {
        "logging_dir": "./logs",
        "automatic_checkpoint_naming": True,
        "total_limit": 5,
        "project_dir": f"checkpoints/{project_dir}",
    }

    if training_args.log_with == "wandb":
        training_args.tracker_kwargs = {
            "wandb": {
                "name": f"{unique_id}_{training_args.reward_fn}_{training_args.gradient_estimation_strategy}_seed_{training_args.seed}"
                }
        }
    elif training_args.log_with == "tensorboard":
        # tensorboard 通常不需要 init_kwargs，log_dir 已经在 project_kwargs 的 logging_dir 中设置了
        # 如果一定要设置，应该使用空字典或者不包含 log_dir
        training_args.tracker_kwargs = {}  # 或者 {"tensorboard": {}}
    prompt_fn = getattr(prompts_file, training_args.prompt_fn)
    
    pipeline = DiffusionPipeline(
        script_args.pretrained_model,
        pretrained_model_revision=script_args.pretrained_revision,
        use_lora=script_args.use_lora,
    )
    print(training_args.log_with)
    trainer = RLR_Trainer(
        training_args,
        prompt_fn,
        pipeline,
        image_samples_hook=image_outputs_logger,
    )
    trainer.train()