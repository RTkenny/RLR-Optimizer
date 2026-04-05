import os
import hpsv2
import torch
import numpy as np
import torchvision
from aesthetic_scorer import AestheticScorerDiff, ImageReward
from transformers import AutoProcessor, AutoModel
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

def imagereward_loss_fn(inference_dtype=None, device=None, target=None, grad_scale=1.0):
    med_config = '/home/rt/data/model/THUDM/ImageReward/med_config.json'
    model_path= '/home/rt/data/model/THUDM/ImageReward/ImageReward.pt'
    model = ImageReward(device=device, med_config=med_config, dtype=inference_dtype).to(device, dtype=inference_dtype)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict,strict=False)
    model.eval()
    def loss_fn(im_pix, prompts):
        rewards = model.score(prompt=prompts, image=im_pix)
        if target is None: # default maximization
            loss = -1 * rewards
        else:
            # using L1 to keep on same scale
            loss = abs(rewards - target)
        return loss * grad_scale, rewards
    return loss_fn

def hps_loss_fn(inference_dtype=None, device=None):
    model_name = "ViT-H-14"
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        model_name,
        'laion2B-s32B-b79K',
        precision=inference_dtype,
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False
    )    
    
    tokenizer = get_tokenizer(model_name)
    
    link = "https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt"
    import os
    import requests
    from tqdm import tqdm

    # Create the directory if it doesn't exist
    os.makedirs(os.path.expanduser('~/.cache/hpsv2'), exist_ok=True)
    # checkpoint_path = f"{os.path.expanduser('~')}/.cache/hpsv2/HPS_v2_compressed.pt"
    checkpoint_path = f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--xswu--HPSv2/snapshots/697403c78157020a1ae59d23f111aa58ced35b0a/HPS_v2_compressed.pt"

    # Download the file if it doesn't exist
    if not os.path.exists(checkpoint_path):
        response = requests.get(link, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(checkpoint_path, 'wb') as file, tqdm(
            desc="Downloading HPS_v2_compressed.pt",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)
    
    
    # force download of model via score
    hpsv2.score([], "")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer(model_name)
    model = model.to(device, dtype=inference_dtype)
    model.eval()

    target_size = 224
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
        
    def loss_fn(im_pix, prompts):    
        im_pix = ((im_pix / 2) + 0.5).clamp(0, 1) 
        x_var = torchvision.transforms.Resize(target_size)(im_pix)
        x_var = normalize(x_var).to(im_pix.dtype)        
        caption = tokenizer(prompts)
        caption = caption.to(device)
        outputs = model(x_var, caption)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits = image_features @ text_features.T
        scores = torch.diagonal(logits)
        loss = 1.0 - scores
        return  loss, scores
    
    return loss_fn
    

def aesthetic_loss_fn(aesthetic_target=None,
                     grad_scale=0,
                     device=None,
                     torch_dtype=None):
    
    target_size = 224
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
    scorer = AestheticScorerDiff(dtype=torch_dtype).to(device, dtype=torch_dtype)
    scorer.requires_grad_(False)
    target_size = 224
    def loss_fn(im_pix_un):
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1) 
        im_pix = torchvision.transforms.Resize(target_size)(im_pix)
        im_pix = normalize(im_pix).to(im_pix_un.dtype)
        rewards = scorer(im_pix)
        if aesthetic_target is None: # default maximization
            loss = -1 * rewards
        else:
            # using L1 to keep on same scale
            loss = abs(rewards - aesthetic_target)
        return loss * grad_scale, rewards
    return loss_fn

def aesthetic_hps_loss_fn(aesthetic_target=None,
                     grad_scale=0,
                     inference_dtype=None, 
                     device=None, 
                     hps_version="v2.0"):
    '''
    Args:
        aesthetic_target: float, the target value of the aesthetic score. it is 10 in this experiment
        grad_scale: float, the scale of the gradient. it is 0.1 in this experiment
        inference_dtype: torch.dtype, the data type of the model.
        device: torch.device, the device to run the model.
        hps_version: str, the version of the HPS model. It is "v2.0" or "v2.1" in this experiment.

    Returns:
        loss_fn: function, the loss function of a combination of aesthetic and HPS reward function.
    '''
    # HPS
    model_name = "ViT-H-14"
    
    model, preprocess_train, preprocess_val = create_model_and_transforms(
            model_name,
            'laion2B-s32B-b79K',
            precision=inference_dtype,
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )    
    
    # tokenizer = get_tokenizer(model_name)
    
    if hps_version == "v2.0":   # if there is a error, please download the model manually and set the path
        checkpoint_path = f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--xswu--HPSv2/snapshots/697403c78157020a1ae59d23f111aa58ced35b0a/HPS_v2_compressed.pt"
    else:   # hps_version == "v2.1"
        checkpoint_path = f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--xswu--HPSv2/snapshots/697403c78157020a1ae59d23f111aa58ced35b0a/HPS_v2.1_compressed.pt"
    # force download of model via score
    hpsv2.score([], "", hps_version=hps_version)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer(model_name)
    model = model.to(device, dtype=inference_dtype)
    model.eval()

    target_size =  (224, 224)
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
    # Aesthetic
    scorer = AestheticScorerDiff(dtype=inference_dtype).to(device, dtype=inference_dtype)
    scorer.requires_grad_(False)
    
    def loss_fn(im_pix_un, prompts):
        # Aesthetic
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1)
        im_pix = torchvision.transforms.Resize(target_size)(im_pix)
        im_pix = normalize(im_pix).to(im_pix_un.dtype)

        aesthetic_rewards = scorer(im_pix)
        if aesthetic_target is None: # default maximization
            aesthetic_loss = -1 * aesthetic_rewards
        else:
            # using L1 to keep on same scale
            aesthetic_loss = abs(aesthetic_rewards - aesthetic_target)
        aesthetic_loss = aesthetic_loss.mean() * grad_scale
        aesthetic_rewards = aesthetic_rewards.mean()

        # HPS
        caption = tokenizer(prompts)
        caption = caption.to(device)
        outputs = model(im_pix, caption)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits = image_features @ text_features.T
        scores = torch.diagonal(logits)
        hps_loss = abs(1.0 - scores)
        hps_loss = hps_loss.mean()
        hps_rewards = scores.mean()

        loss = (1.3 * aesthetic_loss + hps_loss) /2  # 1.5 is a hyperparameter. Set it to 1.5 because experimentally hps_loss is 1.5 times larger than aesthetic_loss
        rewards = (aesthetic_rewards + 15 * hps_rewards) / 2    # 15 is a hyperparameter. Set it to 15 because experimentally aesthetic_rewards is 15 times larger than hps_reward
        return loss, rewards
    
    return loss_fn

def pick_score_loss_fn(inference_dtype=None, device=None):
    '''
    Args:
        inference_dtype: torch.dtype, the data type of the model.
        device: torch.device, the device to run the model.

    Returns:
        loss_fn: function, the loss function of the PickScore reward function.
    '''
    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"
    processor = AutoProcessor.from_pretrained(processor_name_or_path, torch_dtype=inference_dtype)
    model = AutoModel.from_pretrained(model_pretrained_name_or_path, torch_dtype=inference_dtype).eval().to(device)
    model.requires_grad_(False)

    def loss_fn(im_pix_un, prompts):    # im_pix_un: b,c,h,w
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1)

        # reproduce the pick_score preprocessing
        im_pix = im_pix * 255   # b,c,h,w

        if im_pix.shape[2] < im_pix.shape[3]:
            height = 224
            width = im_pix.shape[3] * height // im_pix.shape[2]    # keep the aspect ratio, so the width is w * 224/h
        else:
            width = 224
            height = im_pix.shape[2] * width // im_pix.shape[3]    # keep the aspect ratio, so the height is h * 224/w

        # interpolation and antialiasing should be the same as below
        im_pix = torchvision.transforms.Resize((height, width), 
                                               interpolation=torchvision.transforms.InterpolationMode.BICUBIC, 
                                               antialias=True)(im_pix)
        im_pix = im_pix.permute(0, 2, 3, 1)  # b,c,h,w -> (b,h,w,c)
        # crop the center 224x224
        startx = width//2 - (224//2)
        starty = height//2 - (224//2)
        im_pix = im_pix[:, starty:starty+224, startx:startx+224, :]
        # do rescale and normalize as CLIP
        im_pix = im_pix * 0.00392156862745098   # rescale factor
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
        im_pix = (im_pix - mean) / std
        im_pix = im_pix.permute(0, 3, 1, 2)  # BHWC -> BCHW
        
        text_inputs = processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)

        
        # embed
        image_embs = model.get_image_features(pixel_values=im_pix)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        # score
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        loss = abs(1.0 - scores / 100.0)
        return loss.mean(), scores.mean()
    
    return loss_fn