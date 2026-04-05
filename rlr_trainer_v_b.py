import os
import time
import datetime
import textwrap
import random
import torch
import numpy as np
from collections import defaultdict
from typing import Any, Callable, List, Optional, Tuple, Union
from warnings import warn
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from transformers import is_wandb_available
from sd_pipeline import DiffusionPipeline
from rlr_config import RLR_Config
from trl.trainer import BaseTrainer, DDPOTrainer, AlignPropTrainer
from loss_fn import hps_loss_fn, aesthetic_loss_fn, aesthetic_hps_loss_fn, pick_score_loss_fn, imagereward_loss_fn

if is_wandb_available():
    import wandb

logger = get_logger(__name__)


class RLR_Trainer(BaseTrainer):
    """
    The RLRTrainer is a half-order optimization framework to train a Stable Diffusion model.
    As of now only Stable Diffusion based pipelines are supported

    Attributes:
        config (`RLRConfig`):
            Configuration object for RLRTrainer. Check the documentation of `PPOConfig` for more details.
        reward_function (`Callable[[torch.Tensor, Tuple[str], Tuple[Any]], torch.Tensor]`):
            Reward function to be used
        prompt_function (`Callable[[], Tuple[str, Any]]`):
            Function to generate prompts to guide model
        sd_pipeline (`DiffusionPipeline`):
            Stable Diffusion pipeline to be used for training.
        image_samples_hook (`Optional[Callable[[Any, Any, Any], Any]]`):
            Hook to be called to log images
    """

    _tag_names = ["trl", "RLR"]

    def __init__(
        self,
        config: RLR_Config,
        prompt_function: Callable[[], Tuple[str, Any]],
        sd_pipeline: DiffusionPipeline,
        image_samples_hook: Optional[Callable[[Any, Any, Any], Any]] = None,
    ):
        if image_samples_hook is None:
            warn("No image_samples_hook provided; no images will be logged")

        self.prompt_fn = prompt_function
        
        self.config = config
        self.image_samples_callback = image_samples_hook

        accelerator_project_config = ProjectConfiguration(**self.config.project_kwargs)

        if self.config.resume_from:
            self.config.resume_from = os.path.normpath(os.path.expanduser(self.config.resume_from))
            if "checkpoint_" not in os.path.basename(self.config.resume_from):
                # get the most recent checkpoint in this directory
                checkpoints = list(
                    filter(
                        lambda x: "checkpoint_" in x,
                        os.listdir(self.config.resume_from),
                    )
                )
                if len(checkpoints) == 0:
                    raise ValueError(f"No checkpoints found in {self.config.resume_from}")
                checkpoint_numbers = sorted([int(x.split("_")[-1]) for x in checkpoints])
                self.config.resume_from = os.path.join(
                    self.config.resume_from,
                    f"checkpoint_{checkpoint_numbers[-1]}",
                )

                accelerator_project_config.iteration = checkpoint_numbers[-1] + 1

        self.num_train_timesteps = int(self.config.sample_num_steps * self.config.timestep_fraction)
        
        if self.config.gradient_estimation_strategy == "RL":
            # gradient_accumulation_steps = self.config.train_gradient_accumulation_steps * (self.num_train_timesteps // self.config.chain_len + self.num_train_timesteps % self.config.chain_len)
            gradient_accumulation_steps = self.config.train_gradient_accumulation_steps * self.num_train_timesteps
            
        elif self.config.gradient_estimation_strategy == "RLR":
            # gradient_accumulation_steps = self.config.train_zo_sample_budget
            gradient_accumulation_steps = self.config.train_gradient_accumulation_steps* self.config.chain_len+1
        else:
            gradient_accumulation_steps = self.config.train_gradient_accumulation_steps
        
        self.accelerator = Accelerator(
            log_with=self.config.log_with,
            mixed_precision=self.config.mixed_precision,
            project_config=accelerator_project_config,
            # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
            # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
            # the total number of optimizer steps to accumulate across.
            gradient_accumulation_steps=gradient_accumulation_steps,
            **self.config.accelerator_kwargs,
        )

        is_using_tensorboard = config.log_with is not None and config.log_with == "tensorboard"
        config_dict = config.to_dict()
        config_dict['checkpoints_dir'] = self.config.project_kwargs['project_dir']

        if self.accelerator.is_main_process:
            print(f"[DEBUG] Before init_trackers:")
            print(f"  - is_main_process: {self.accelerator.is_main_process}")
            print(f"  - log_with: {self.config.log_with}")
            print(f"  - tracker_project_name: {self.config.tracker_project_name}")
            print(f"  - tracker_kwargs: {self.config.tracker_kwargs}")
            print(f"  - is_using_tensorboard: {is_using_tensorboard}")
            print(f"  - trackers before: {self.accelerator.trackers}")
            print(f"  - project_config.logging_dir: {accelerator_project_config.logging_dir}")

            tracker_project_name = self.config.tracker_project_name
            
            # 对于 tensorboard，确保 init_kwargs 显式指定
            if is_using_tensorboard:
                base_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = base_ts
                run_index = 1
                logging_root = accelerator_project_config.logging_dir
                while os.path.exists(os.path.join(logging_root, tracker_project_name, run_name)):
                    run_name = f"{base_ts}_{run_index}"
                    run_index += 1
                tracker_project_name = os.path.join(tracker_project_name, run_name)
                # tensorboard 需要显式指定，即使为空字典
                init_kwargs = {"tensorboard": {}} if not self.config.tracker_kwargs else self.config.tracker_kwargs
            else:
                init_kwargs = self.config.tracker_kwargs
            
            print(f"  - final init_kwargs: {init_kwargs}")
            print(f"  - final tracker_project_name: {tracker_project_name}")
            
            try:
                self.accelerator.init_trackers(
                    tracker_project_name,
                    config=dict(alignprop_trainer_config=config_dict)
                    if not is_using_tensorboard
                    else config.to_dict(),
                    init_kwargs=init_kwargs,
                )
                print(f"[DEBUG] After init_trackers:")
                print(f"  - trackers: {self.accelerator.trackers}")
                print(f"  - len(trackers): {len(self.accelerator.trackers)}")
                if len(self.accelerator.trackers) > 0:
                    print(f"  - tracker type: {type(self.accelerator.trackers[0])}")
                    print(f"  - tracker: {self.accelerator.trackers[0]}")
                else:
                    print(f"[WARNING] No trackers were created. Image logging will be skipped.")
            except Exception as e:
                print(f"[ERROR] init_trackers failed with exception: {e}")
                import traceback
                traceback.print_exc()

        logger.info(f"\n{config}")

        set_seed(self.config.seed, device_specific=True)

        self.sd_pipeline = sd_pipeline

        self.sd_pipeline.set_progress_bar_config(
            position=1,
            disable=not self.accelerator.is_local_main_process,
            leave=False,
            desc="Timestep",
            dynamic_ncols=True,
        )

        # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        if self.accelerator.mixed_precision == "fp16":
            inference_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            inference_dtype = torch.bfloat16
        else:
            inference_dtype = torch.float32

        self.sd_pipeline.vae.to(self.accelerator.device, dtype=inference_dtype)
        self.sd_pipeline.text_encoder.to(self.accelerator.device, dtype=inference_dtype)
        self.sd_pipeline.unet.to(self.accelerator.device, dtype=inference_dtype)

        trainable_layers = self.sd_pipeline.get_trainable_layers()
        # print(len(trainable_layers))
        # print(trainable_layers)

        self.accelerator.register_save_state_pre_hook(self._save_model_hook)
        self.accelerator.register_load_state_pre_hook(self._load_model_hook)

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        self.optimizer = self._setup_optimizer(
            trainable_layers.parameters() if not isinstance(trainable_layers, list) else trainable_layers
        )

        self.neg_prompt_embed = self.sd_pipeline.text_encoder(
            self.sd_pipeline.tokenizer(
                [""] if self.config.negative_prompts is None else self.config.negative_prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.sd_pipeline.tokenizer.model_max_length,
            ).input_ids.to(self.accelerator.device)
        )[0]

        # NOTE: for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
        # more memory
        self.autocast = self.sd_pipeline.autocast or self.accelerator.autocast

        if hasattr(self.sd_pipeline, "use_lora") and self.sd_pipeline.use_lora:
            unet, self.optimizer = self.accelerator.prepare(trainable_layers, self.optimizer)
            self.trainable_layers = list(filter(lambda p: p.requires_grad, unet.parameters()))
        else:
            self.trainable_layers, self.optimizer = self.accelerator.prepare(trainable_layers, self.optimizer)
        
        if self.config.reward_fn=='hps':
            self.loss_fn = hps_loss_fn(inference_dtype, self.accelerator.device)
        elif self.config.reward_fn=='aesthetic': # easthetic
            self.loss_fn = aesthetic_loss_fn(
                grad_scale=self.config.grad_scale,
                aesthetic_target=self.config.aesthetic_target,
                torch_dtype = inference_dtype,
                device = self.accelerator.device
            )
        elif self.config.reward_fn=='hps_aesthetic':
            self.loss_fn = aesthetic_hps_loss_fn(
                aesthetic_target=self.config.aesthetic_target,
                grad_scale=self.config.grad_scale,
                inference_dtype = inference_dtype,
                device = self.accelerator.device
            )
        elif self.config.reward_fn=='pickscore':
            self.loss_fn = pick_score_loss_fn(
                inference_dtype = inference_dtype,
                device = self.accelerator.device
            )
        elif self.config.reward_fn=='imagereward':
            self.loss_fn = imagereward_loss_fn(
                inference_dtype = inference_dtype,
                device = self.accelerator.device
            )
        else:
            raise NotImplementedError
        if config.resume_from:
            logger.info(f"Resuming from {config.resume_from}")
            self.accelerator.load_state(config.resume_from)
            self.first_epoch = int(config.resume_from.split("_")[-1]) + 1
        else:
            self.first_epoch = 0
        
        self.eval_prompts, self.eval_prompt_metadata = zip(*[self.prompt_fn() for _ in range(config.eval_batch_size)])

    def reward_fn_RL(self, images, prompts, prompt_metadata):
        """
        Compute the reward for a given image and prompt

        Args:
            images (torch.Tensor):
                The images to compute the reward for, shape: [batch_size, 3, height, width]
            prompts (Tuple[str]):
                The prompts to compute the reward for
            prompt_metadata (Tuple[Any]):
                The metadata associated with the prompts

        Returns:
            reward (torch.Tensor), reward_metadata (Any)
        """
        if "hps" in self.config.reward_fn:
            loss, rewards = self.loss_fn(images, prompts)
            return rewards, {}
        elif self.config.reward_fn == "aesthetic":
            loss, rewards = self.loss_fn(images)
            return rewards, {}
        else:
            raise NotImplementedError

    def compute_rewards(self, prompt_image_pairs, is_async=False):
        rewards = []
        for images, prompts, prompt_metadata in prompt_image_pairs:
            reward, reward_metadata = self.reward_fn_RL(images, prompts, prompt_metadata)
            rewards.append(
                (
                    torch.as_tensor(reward, device=self.accelerator.device),
                    reward_metadata,
                )
            )
        return zip(*rewards)
    
    def perturb_all_params(self, random_seed=None, scaling_factor=1):
        """
        Perturb the all parameters with random vector z.
        Input: 
        - random_seed: random seed for in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * self.config.zo_eps

    def zo_forward(self, model, prompts, prompt_metadata, retain_graph=False):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()
        batch_size = len(prompts) # self.config.train_batch_size
        sample_neg_prompt_embeds = self.neg_prompt_embed.repeat(batch_size, 1, 1)
        if not retain_graph:
            with torch.inference_mode():
                sd_output, prompt_ids, prompt_embeds = self._inference_steps(prompts, sample_neg_prompt_embeds, with_grad=False)
                if "hps" in self.config.reward_fn or self.config.reward_fn=='imagereward' or self.config.reward_fn=='pickscore':
                    loss, rewards = self.loss_fn(sd_output.images, prompts)
                else:
                    loss, rewards = self.loss_fn(sd_output.images)
        else:
            sd_output, prompt_ids, prompt_embeds = self._inference_steps(prompts, sample_neg_prompt_embeds, with_grad=False)
            if "hps" in self.config.reward_fn:
                loss, rewards = self.loss_fn(sd_output.images, prompts)
            else:
                loss, rewards = self.loss_fn(sd_output.images)
        model.train()
        print(f'log_probs:{sd_output.log_probs}')
        return loss.mean().detach(), rewards

    def zo_backward(self, target_name=None):
        torch.manual_seed(self.zo_random_seed)     

        for name, param in self.named_parameters_to_optim:
            if target_name is not None and target_name != name:
                continue
            # Resample z
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            if param.grad is None:
                param.grad = self.projected_grad * z
            else:
                param.grad += self.projected_grad * z

    def calculate_loss(self, latents, timesteps, next_latents, log_probs, advantages, embeds):
        """
        Calculate the loss for a batch of an unpacked sample

        Args:
            latents (torch.Tensor):
                The latents sampled from the diffusion model, shape: [batch_size, num_channels_latents, height, width]
            timesteps (torch.Tensor):
                The timesteps sampled from the diffusion model, shape: [batch_size]
            next_latents (torch.Tensor):
                The next latents sampled from the diffusion model, shape: [batch_size, num_channels_latents, height, width]
            log_probs (torch.Tensor):
                The log probabilities of the latents, shape: [batch_size]
            advantages (torch.Tensor):
                The advantages of the latents, shape: [batch_size]
            embeds (torch.Tensor):
                The embeddings of the prompts, shape: [2*batch_size or batch_size, ...]
                Note: the "or" is because if train_cfg is True, the expectation is that negative prompts are concatenated to the embeds

        Returns:
            loss (torch.Tensor), approx_kl (torch.Tensor), clipfrac (torch.Tensor)
            (all of these are of shape (1,))
        """
        temp_latents = None
        temp_eta = 0.0
        temp_chain_len = timesteps.shape[1]
        for i in range(temp_chain_len):
            # if i == self.config.chain_len - 1 or timesteps[0].cpu().numpy() == 0:
            #     temp_eta = self.config.sample_eta
            #     temp_latents = next_latents

            with self.autocast():
                if self.config.train_cfg:
                    noise_pred = self.sd_pipeline.unet(
                        torch.cat([latents[:,i]] * 2),
                        torch.cat([timesteps[:,i]] * 2),
                        embeds,
                    ).sample # .sample is equal to [0] ？
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.config.sample_guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                else:
                    noise_pred = self.sd_pipeline.unet(
                        latents[:,i],
                        timesteps[:,i],
                        embeds,
                    ).sample

                # compute the log prob of next_latents given latents under the current model
                scheduler_step_output = self.sd_pipeline.scheduler_step(
                    noise_pred,
                    timesteps[:,i],
                    latents[:,i],
                    eta=self.config.sample_eta,
                    prev_sample=next_latents[:,i], # 这一行会导致一些奇怪的问题 this line causes some strange problems
                )

                timesteps -= 1
                log_prob = scheduler_step_output.log_probs

        advantages = torch.clamp(
            advantages,
            -self.config.train_adv_clip_max,
            self.config.train_adv_clip_max,
        )
        if self.config.gradient_estimation_strategy == "RL":
            ratio = torch.exp(log_prob - log_probs)
            loss = self.loss(advantages, self.config.train_clip_range, ratio)
            approx_kl = 0.5 * torch.mean((log_prob - log_probs) ** 2)
            clipfrac = torch.mean((torch.abs(ratio - 1.0) > self.config.train_clip_range).float())

        return loss, approx_kl, clipfrac

    def loss(
        self,
        advantages: torch.Tensor,
        clip_range: float,
        ratio: torch.Tensor,
    ):
        unclipped_loss = -advantages * ratio
        # return torch.mean(unclipped_loss)
        clipped_loss = -advantages * torch.clamp(
            ratio,
            1.0 - clip_range,
            1.0 + clip_range,
        )
        return torch.mean(torch.maximum(unclipped_loss, clipped_loss))
    
    def _train_batched_samples(self, inner_epoch, epoch, global_step, batched_samples):
        """
        Train on a batch of samples. Main training segment

        Args:
            inner_epoch (int): The current inner epoch
            epoch (int): The current epoch
            global_step (int): The current global step
            batched_samples (List[Dict[str, torch.Tensor]]): The batched samples to train on

        Side Effects:
            - Model weights are updated
            - Logs the statistics to the accelerator trackers.

        Returns:
            global_step (int): The updated global step
        """
        info = defaultdict(list)
        for _i, sample in enumerate(batched_samples):
            if self.config.train_cfg:
                # concat negative prompts to sample prompts to avoid two forward passes
                embeds = torch.cat([sample["negative_prompt_embeds"], sample["prompt_embeds"]])
            else:
                embeds = sample["prompt_embeds"]

            for j in range(self.num_train_timesteps):
                with self.accelerator.accumulate(self.sd_pipeline.unet):
                    
                    if np.random.rand() < 0.5:
                        continue
                    if j % self.config.chain_len != 0:
                        fake_loss = torch.tensor(0.0, requires_grad=True, device=self.accelerator.device)
                        self.accelerator.backward(fake_loss)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        continue

                    loss, approx_kl, clipfrac = self.calculate_loss(
                        sample["latents"][:, j:j+self.config.chain_len],
                        sample["timesteps"][:, j:j+self.config.chain_len],
                        sample["next_latents"][:, j:j+self.config.chain_len],
                        sample["log_probs"][:, j+self.config.chain_len-1],
                        sample["advantages"],
                        embeds,
                    )
                    
                    # print(f"loss: {loss}, approx_kl: {approx_kl}, clipfrac: {clipfrac}")
                    if loss.isnan():
                        fake_loss = torch.tensor(0.0, requires_grad=True, device=self.accelerator.device)
                        self.accelerator.backward(fake_loss)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        continue
                    info["approx_kl"].append(approx_kl)
                    info["clipfrac"].append(clipfrac)
                    info["loss"].append(loss)

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.trainable_layers.parameters()
                            if not isinstance(self.trainable_layers, list)
                            else self.trainable_layers,
                            self.config.train_max_grad_norm,
                        )
            
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
        # Checks if the accelerator has performed an optimization step behind the scenes
        if self.accelerator.sync_gradients:
            # log training-related stuff
            info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
            info = self.accelerator.reduce(info, reduction="mean")
            info.update({"epoch": epoch, "inner_epoch": inner_epoch})
            self.accelerator.log(info, step=global_step)
            global_step += 1
            print(f"log info: {info}")
            info = defaultdict(list)
        # else:
        #     # self.accelerator.wait_for_everyone()
        #     raise ValueError("Optimization step should have been performed by this point. Please check calculated gradient accumulation settings.")

        return global_step

    def step(self, epoch: int, global_step: int):
        """
        Perform a single step of training.

        Args:
            epoch (int): The current epoch.
            global_step (int): The current global step.

        Side Effects:
            - Model weights are updated
            - Logs the statistics to the accelerator trackers.
            - If `self.image_samples_callback` is not None, it will be called with the prompt_image_pairs, global_step, and the accelerator tracker.

        Returns:
            global_step (int): The updated global step.
        """
        info = defaultdict(list)
        print(f"Epoch: {epoch}, Global Step: {global_step}")

        self.sd_pipeline.unet.train()
        
        if self.config.gradient_estimation_strategy == "RL":  
            ## version 2 from trl ##
            samples, prompt_image_data = self._generate_samples(
                iterations=self.config.sample_num_batches_per_epoch,
                batch_size=self.config.sample_batch_size,
                with_grad=False
            )

            # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
            samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
            rewards, rewards_metadata = self.compute_rewards(
                prompt_image_data, is_async=self.config.async_reward_computation
            )

            for i, image_data in enumerate(prompt_image_data):
                image_data.extend([rewards[i], rewards_metadata[i]])

            # if self.image_samples_callback is not None and self.accelerator.is_main_process: # add "and self.accelerator.is_main_process"
            #     self.image_samples_callback(prompt_image_data, global_step, self.accelerator.trackers[0])

            rewards = torch.cat(rewards)
            rewards = self.accelerator.gather(rewards).detach().cpu().numpy()

            self.accelerator.log(
                {
                    "reward": rewards,
                    "epoch": epoch,
                    "reward_mean": rewards.mean(),
                    "reward_std": rewards.std(),
                },
                step=global_step,
            )

            if self.config.per_prompt_stat_tracking: # no implementation for per prompt stat tracking yet, so this block will not be executed for now
                # gather the prompts across processes
                prompt_ids = self.accelerator.gather(samples["prompt_ids"]).cpu().numpy()
                prompts = self.sd_pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
                advantages = self.stat_tracker.update(prompts, rewards)
            else:
                advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            # ungather advantages;  keep the entries corresponding to the samples on this process
            samples["advantages"] = (
                torch.as_tensor(advantages)
                .reshape(self.accelerator.num_processes, -1)[self.accelerator.process_index]
                .to(self.accelerator.device)
            )

            del samples["prompt_ids"]

            total_batch_size, num_timesteps = samples["timesteps"].shape
            print(f'batch size:{total_batch_size}')
            print(f'timesteps:{num_timesteps}')
            print(f'log_probs:{samples["log_probs"]}')
            for key in samples.keys():
                print(f'{key}: {samples[key].shape}')

            for inner_epoch in range(self.config.train_num_inner_epochs):
                # shuffle samples along batch dimension
                perm = torch.randperm(total_batch_size, device=self.accelerator.device)
                samples = {k: v[perm] for k, v in samples.items()}
                
                # Create perms that keep subchain timesteps together
                chain_len = self.config.chain_len
                num_subchains = num_timesteps // chain_len
                
                # First create permutations for the subchain indices
                subchain_perms = torch.stack([
                    torch.randperm(num_subchains, device=self.accelerator.device) 
                    for _ in range(total_batch_size)
                ])
                
                # Expand each subchain index to its constituent timesteps
                perms = torch.zeros((total_batch_size, num_timesteps), dtype=torch.long, device=self.accelerator.device)
                for i in range(num_subchains):
                    perms[:, i*chain_len:(i+1)*chain_len] = (subchain_perms[:, i:i+1] * chain_len).repeat(1, chain_len) + torch.arange(chain_len, device=self.accelerator.device)
                    
                # Handle remaining timesteps if num_timesteps is not divisible by chain_len
                remaining = num_timesteps % chain_len
                if remaining > 0:
                    perms[:, -remaining:] = torch.arange(num_timesteps-remaining, num_timesteps, device=self.accelerator.device)

                # # shuffle along time dimension independently for each sample
                # perms = torch.stack(
                #     [torch.randperm(num_timesteps, device=self.accelerator.device) for _ in range(total_batch_size)]
                # )

                # perms = torch.stack(
                #     [torch.arange(num_timesteps, device=self.accelerator.device) for _ in range(total_batch_size)]
                # )

                for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                    samples[key] = samples[key][
                        torch.arange(total_batch_size, device=self.accelerator.device)[:, None],
                        perms,
                    ]

                original_keys = samples.keys()
                original_values = samples.values()
                # rebatch them as user defined train_batch_size is different from sample_batch_size
                reshaped_values = [v.reshape(-1, self.config.train_batch_size, *v.shape[1:]) for v in original_values]

                # Transpose the list of original values
                transposed_values = zip(*reshaped_values)
                # Create new dictionaries for each row of transposed values
                samples_batched = [dict(zip(original_keys, row_values)) for row_values in transposed_values]

                self.sd_pipeline.unet.train()
                global_step = self._train_batched_samples(inner_epoch, epoch, global_step, samples_batched)
                # ensure optimization step at the end of the inner epoch
                if not self.accelerator.sync_gradients:
                    raise ValueError(
                        "Optimization step should have been performed by this point. Please check calculated gradient accumulation settings."
                    )
            
            if self.image_samples_callback is not None and global_step % self.config.log_image_freq == 0 and self.accelerator.is_main_process:
                print("Logging images")
                # Fix the random seed for reproducibility
                torch.manual_seed(self.config.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.config.seed)
                _, prompt_image_pairs = self._generate_samples(
                        iterations = 1,
                        batch_size=8, # self.config.train_batch_size
                        with_grad=False, 
                        prompts=self.eval_prompts
                    )
                self.image_samples_callback(prompt_image_pairs, global_step, self.accelerator.trackers[0])
                
        
        elif self.config.gradient_estimation_strategy == "RLR":
            
            prompts, prompt_metadata = self._collect_data(self.config.train_batch_size)

            for _ in range(self.config.train_gradient_accumulation_steps):
                with self.accelerator.accumulate(self.sd_pipeline.unet), self.autocast(), torch.enable_grad(): # torch.no_grad or torch.enable_grad
                    samples, prompt_image_pairs = self._generate_samples(
                                iterations = 1,
                                batch_size=self.config.train_batch_size,
                                with_grad=True,
                                prompts=prompts
                            )
                    prompt_image_data = {}
                    prompt_image_data["images"] = prompt_image_pairs[0][0]
                    prompt_image_data["prompts"] = prompt_image_pairs[0][1]
                    prompt_image_data["prompt_metadata"] = prompt_image_pairs[0][2]

                    print("Samples generated")
                    samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

                    if "hps" in self.config.reward_fn or self.config.reward_fn=='imagereward' or self.config.reward_fn=='pickscore':
                        loss, rewards = self.loss_fn(prompt_image_data["images"], prompt_image_data["prompts"])
                    else:
                        loss, rewards = self.loss_fn(prompt_image_data["images"])

                    rewards = self.accelerator.gather(rewards).detach().cpu().numpy()
                    rewards_mean = float(rewards.mean())
                    rewards_std = float(rewards.std())
                    rewards_std = rewards_std if rewards_std > 0.0 else 1.0
                    normalized_rewards = (rewards - rewards_mean) / rewards_std
                    # normalized_rewards = np.clip(
                    #     normalized_rewards,
                    #     -self.config.train_adv_clip_max,
                    #     self.config.train_adv_clip_max,
                    # )

                    loss = loss.mean()
                    FO_loss = loss * self.config.loss_coeff
                    loss_dict = {}
                    loss_dict['FO_loss'] = FO_loss

                    # option 2: separate FO and HO backward
                    self.accelerator.backward(FO_loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    loss_dict['HO_loss'] = torch.tensor(0.0, dtype=torch.float32, device=self.accelerator.device)
                    # self.accelerator.backward(FO_loss)
                    # self.optimizer.step()
                    # self.optimizer.zero_grad()

                    for i in range(self.config.chain_len):
                        target_step = np.random.randint(self.config.sample_num_steps)
                        print(f'i:{i}')

                        if self.config.train_cfg:
                            embeds = torch.cat([samples["negative_prompt_embeds"], samples["prompt_embeds"]])
                            noise_pred = self.sd_pipeline.unet(
                                torch.cat([samples["latents"][:, target_step]] * 2),
                                torch.cat([samples["timesteps"][:, target_step]] * 2),
                                embeds,
                            ).sample # .sample is equal to [0] ？
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + self.config.sample_guidance_scale * (
                                noise_pred_text - noise_pred_uncond
                            )
                        else:
                            embeds = samples["prompt_embeds"]
                            noise_pred = self.sd_pipeline.unet(
                                samples["latents"][:,target_step],
                                samples["timesteps"][:,target_step],
                                embeds,
                            ).sample

                        scheduler_step_output = self.sd_pipeline.scheduler_step(
                            noise_pred,
                            samples["timesteps"][:,target_step],
                            samples["latents"][:,target_step],
                            eta=self.config.sample_eta,
                            prev_sample=samples["next_latents"][:,target_step], # 这一行会导致一些奇怪的问题 this line causes some strange problems
                        )

                        log_prob = scheduler_step_output.log_probs
                        old_log_prob = samples["log_probs"][:, target_step].detach()
                        log_ratio = log_prob - old_log_prob
                        log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
                        ratio = torch.exp(log_ratio)
                        advantage = torch.tensor(normalized_rewards, device=self.accelerator.device)
                        HO_loss = self.loss(advantage, self.config.train_clip_range, ratio) * self.config.ho_loss_coeff
                        print(f'HO_loss: {HO_loss}')
                        print(
                            "HO debug | reward_mean: %.6f reward_std: %.6f ratio_min: %.6f ratio_max: %.6f ratio_mean: %.6f"
                            % (
                                rewards_mean,
                                rewards_std,
                                ratio.min().item(),
                                ratio.max().item(),
                                ratio.mean().item(),
                            )
                        )

                        #option 2: separate FO and HO backward
                        if HO_loss.isnan():
                            HO_loss = torch.tensor(0.0, dtype=torch.float32, device=self.accelerator.device)
                        # time.sleep(5)
                        self.accelerator.backward(HO_loss)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        # import ipdb; ipdb.set_trace()
                        # HO_loss = torch.mean(-log_prob * torch.tensor(rewards, device=self.accelerator.device))
                        # self.accelerator.backward(HO_loss)
                        if loss_dict.get('HO_loss') is None:
                            loss_dict['HO_loss'] = HO_loss
                        else:
                            loss_dict['HO_loss'] += HO_loss

                    # # # Option 1: Combine losses (recommended)
                    # if loss_dict.get('HO_loss') is not None and loss_dict['HO_loss'].isnan():
                    #     loss_dict['HO_loss'] = torch.tensor(0.0, dtype=torch.float32, device=self.accelerator.device)
                    # total_loss = loss_dict['FO_loss'] + loss_dict['HO_loss']
                    # self.accelerator.backward(total_loss)
                    
                    print(f'loss_dict:{loss_dict}')
                        
                    info["reward_mean"].append(rewards.mean())
                    info["reward_std"].append(rewards.std())
                    total_loss = loss_dict['FO_loss'] + loss_dict['HO_loss']
                    info["loss"].append(total_loss.item())

                    # self.optimizer.step()
                    # self.optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if self.accelerator.sync_gradients:
                # log training-related stuff
                info = {k: torch.mean(torch.tensor(v)) for k, v in info.items()}
                info.update({"epoch": epoch})
                self.accelerator.log(info, step=global_step)
                global_step += 1
                info = defaultdict(list)
            else:
                raise ValueError(
                    "Optimization step should have been performed by this point. Please check calculated gradient accumulation settings."
                )
            if self.image_samples_callback is not None and global_step % self.config.log_image_freq == 0 and self.accelerator.is_main_process:
                print("Logging images")
                # Fix the random seed for reproducibility
                torch.manual_seed(self.config.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.config.seed)
                _, prompt_image_pairs = self._generate_samples(
                        iterations = 1,
                        batch_size=8, # self.config.train_batch_size
                        with_grad=False, 
                        prompts=self.eval_prompts
                    )
                self.image_samples_callback(prompt_image_pairs, global_step, self.accelerator.trackers[0])
                seed = random.randint(0, 100)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)  
                    
        else:
            # BP version
            for _ in range(self.config.train_gradient_accumulation_steps):
                with self.accelerator.accumulate(self.sd_pipeline.unet), self.autocast(), torch.enable_grad():
                    samples, prompt_image_pairs = self._generate_samples(
                            iterations = 1,
                            batch_size=self.config.train_batch_size,
                            with_grad=True
                        )
                    prompt_image_data = {}
                    prompt_image_data["images"] = prompt_image_pairs[0][0]
                    prompt_image_data["prompts"] = prompt_image_pairs[0][1]
                    prompt_image_data["prompt_metadata"] = prompt_image_pairs[0][2]

                    print("Samples generated")
                    samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

                    if "hps" in self.config.reward_fn or self.config.reward_fn=='imagereward' or self.config.reward_fn=='pickscore':
                            loss, rewards = self.loss_fn(prompt_image_data["images"], prompt_image_data["prompts"])
                    else:
                        loss, rewards = self.loss_fn(prompt_image_data["images"])

                    rewards = self.accelerator.gather(rewards).detach().cpu().numpy()

                    if self.config.gradient_estimation_strategy != "LR":
                        loss = loss.mean()
                        loss = loss * self.config.loss_coeff
                        self.accelerator.backward(loss)

                    else:
                        # have problems, originally i want to implement in the policy gradient style 
                        loss = loss.detach()
                        log_probs = torch.sum(samples["log_probs"], dim=1)
                        loss_target = torch.mean(loss*log_probs) * self.config.loss_coeff
                        self.accelerator.backward(loss_target)
                        loss = loss.mean()

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.trainable_layers.parameters()
                            if not isinstance(self.trainable_layers, list)
                            else self.trainable_layers,
                            self.config.train_max_grad_norm,
                        )

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    info["reward_mean"].append(rewards.mean())
                    info["reward_std"].append(rewards.std())
                    info["loss"].append(loss.item())

            # Checks if the accelerator has performed an optimization step behind the scenes
            if self.accelerator.sync_gradients:
                # log training-related stuff
                info = {k: torch.mean(torch.tensor(v)) for k, v in info.items()}
                info.update({"epoch": epoch})
                self.accelerator.log(info, step=global_step)
                global_step += 1
                info = defaultdict(list)
            else:
                raise ValueError(
                    "Optimization step should have been performed by this point. Please check calculated gradient accumulation settings."
                )
            # Logs generated images
            if self.image_samples_callback is not None and global_step % self.config.log_image_freq == 0 and self.accelerator.is_main_process:
                print("Logging images")
                # Fix the random seed for reproducibility
                torch.manual_seed(self.config.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.config.seed)
                _, prompt_image_pairs = self._generate_samples(
                        iterations = 1,
                        batch_size=self.config.eval_batch_size,
                        with_grad=False, 
                        prompts=self.eval_prompts
                    )
                self.image_samples_callback(prompt_image_pairs, global_step, self.accelerator.trackers[0])
                seed = random.randint(0, 100)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)            

        if epoch != 0 and epoch % self.config.save_freq == 0:
            print("Saving checkpoint")
            self.accelerator.save_state()
        print("Step Done")
        return global_step

    def _setup_optimizer(self, trainable_layers_parameters):
        if self.config.train_use_8bit_adam:
            import bitsandbytes

            optimizer_cls = bitsandbytes.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        return optimizer_cls(
            trainable_layers_parameters,
            lr=self.config.train_learning_rate,
            betas=(self.config.train_adam_beta1, self.config.train_adam_beta2),
            weight_decay=self.config.train_adam_weight_decay,
            eps=self.config.train_adam_epsilon,
        )

    def _save_model_hook(self, models, weights, output_dir):
        self.sd_pipeline.save_checkpoint(models, weights, output_dir)
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def _load_model_hook(self, models, input_dir):
        self.sd_pipeline.load_checkpoint(models, input_dir)
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    def _collect_data(self, batch_size):
        return zip(*[self.prompt_fn() for _ in range(batch_size)])

    def _inference_steps(self, prompts, sample_neg_prompt_embeds, with_grad=True, backprop_strategy=None):
        """
            given a prompt, generate samples from the model
        """

        if backprop_strategy is None:
            backprop_strategy = self.config.gradient_estimation_strategy

        prompt_ids = self.sd_pipeline.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.sd_pipeline.tokenizer.model_max_length,
        ).input_ids.to(self.accelerator.device)
        prompt_embeds = self.sd_pipeline.text_encoder(prompt_ids)[0]

        if with_grad:
            sd_output = self.sd_pipeline.rgb_with_grad(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=sample_neg_prompt_embeds,
                num_inference_steps=self.config.sample_num_steps,
                chain_length=self.config.chain_len,
                guidance_scale=self.config.sample_guidance_scale,
                eta=self.config.sample_eta,
                backprop_strategy=backprop_strategy,
                backprop_kwargs=self.config.backprop_kwargs[backprop_strategy],
                output_type="pt",
            )
        else:
            ## version 1
            # sd_output = self.sd_pipeline(
            #     prompt_embeds=prompt_embeds,
            #     negative_prompt_embeds=sample_neg_prompt_embeds,
            #     num_inference_steps=self.config.sample_num_steps,
            #     guidance_scale=self.config.sample_guidance_scale,
            #     eta=self.config.sample_eta,
            #     output_type="pt",
            # )
            ## version 2
            with torch.no_grad():
                sd_output = self.sd_pipeline.rgb_with_grad(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=sample_neg_prompt_embeds,
                num_inference_steps=self.config.sample_num_steps,
                chain_length=self.config.chain_len,
                guidance_scale=self.config.sample_guidance_scale,
                eta=self.config.sample_eta,
                backprop_strategy=backprop_strategy,
                backprop_kwargs=self.config.backprop_kwargs[backprop_strategy],
                output_type="pt",
            )
                
        return sd_output, prompt_ids, prompt_embeds

    def _generate_samples(self, iterations, batch_size, with_grad=True, prompts=None, backprop_strategy=None):
        """
        Generate samples from the model

        Args:
            batch_size (int): Batch size to use for sampling
            with_grad (bool): Whether the generated RGBs should have gradients attached to it.

        Returns:
            
        """
        # original code
        # samples = {}
        if backprop_strategy is None:
            backprop_strategy = self.config.gradient_estimation_strategy

        samples = []
        prompt_image_pairs = []
        # prompt_image_pairs = {}

        sample_neg_prompt_embeds = self.neg_prompt_embed.repeat(batch_size, 1, 1)

        for _ in range(iterations):
            if prompts is None or iterations != 1:
                prompts, prompt_metadata = self._collect_data(batch_size)
            else:
                prompt_metadata = [{} for _ in range(batch_size)]
            # print(f'prompts: {prompts}')
            sd_output, prompt_ids, prompt_embeds = self._inference_steps(prompts, sample_neg_prompt_embeds, with_grad, backprop_strategy)

            images = sd_output.images
            latents = sd_output.latents
            log_probs = sd_output.log_probs

            latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, ...)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
            # timesteps = self.sd_pipeline.scheduler.timesteps[::self.config.chain_len]
            # timesteps = timesteps.repeat(batch_size, 1)  # (batch_size, num_steps)
            timesteps = self.sd_pipeline.scheduler.timesteps.repeat(batch_size, 1)  # (batch_size, num_steps)

            # print(f'latent shape: {latents.shape}')
            # print(f'log_probs shape: {log_probs.shape}')
            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],  # each entry is the latent before timestep t
                    "next_latents": latents[:, 1:],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "negative_prompt_embeds": sample_neg_prompt_embeds,
                }
            )
            prompt_image_pairs.append([images, prompts, prompt_metadata])
        
        return samples, prompt_image_pairs

    def train(self, epochs: Optional[int] = None):
        """
        Train the model for a given number of epochs
        """
        global_step = 0
        if epochs is None:
            epochs = self.config.num_epochs
        for epoch in range(self.first_epoch, epochs):
            global_step = self.step(epoch, global_step)

    def _save_pretrained(self, save_directory):
        self.sd_pipeline.save_pretrained(save_directory)
        self.create_model_card()
