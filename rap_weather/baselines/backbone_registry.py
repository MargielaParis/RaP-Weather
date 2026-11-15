"""Backbone registry shared by the crop-based baselines (CLIP / ConvNeXt2 / DINOv2).

Baseline groups that use this registry (code name -> paper name):

| Code    | Paper                          | Method                                                        | Training |
|---------|--------------------------------|---------------------------------------------------------------|----------|
| Group 1 | Group 1 (off-the-shelf)        | Region crop -> encode on the fly (traditional CBIR)           | None (frozen) |
| Group 2 | Per-region encoders (App. B)   | Region crop -> encode -> sequence of vectors -> predict t+n   | Predictive training, same InfoNCE + cosine loss as RaP-Weather |

In code Group 2 each image is one vector, not a patch grid. The alpha/tau and
LoRA-rank sweeps live in sensitivity_sweep.py and are not repeated here.

All three backbones are available in `timm`, so one interface covers them.
With `num_classes=0`, forward() returns the pooled feature vector directly, so
no per-architecture pooling logic is needed.

Inputs are the [0,1] tensors from normalize_image(); no per-backbone mean/std
normalization is applied.
"""
import timm
import torch.nn.functional as F

from ..config import device

BACKBONE_SPECS = {
    "CLIP":      {"timm_name": "vit_base_patch32_clip_224.openai",       "input_size": 224},
    "ConvNeXt2": {"timm_name": "convnextv2_tiny.fcmae_ft_in22k_in1k",     "input_size": 224},
    "DINOv2":    {"timm_name": "vit_small_patch14_dinov2.lvd142m",       "input_size": 224},
}


def _create_backbone(timm_name, **extra_kwargs):
    """dynamic_img_size exists only for ViT models (CLIP, DINOv2); conv backbones
    such as ConvNeXt raise TypeError. Try with it first, then retry without."""
    try:
        return timm.create_model(timm_name, pretrained=True, num_classes=0, dynamic_img_size=True, **extra_kwargs)
    except TypeError:
        return timm.create_model(timm_name, pretrained=True, num_classes=0, **extra_kwargs)


_FROZEN_BACKBONE_CACHE = {}


def get_frozen_backbone(name, device=device):
    """Fully frozen backbone for Group 1 (off-the-shelf). Cached and reused."""
    if name in _FROZEN_BACKBONE_CACHE:
        return _FROZEN_BACKBONE_CACHE[name]
    spec = BACKBONE_SPECS[name]
    model = _create_backbone(spec["timm_name"])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    _FROZEN_BACKBONE_CACHE[name] = (model, spec["input_size"], model.num_features)
    print(f"[{name}] loaded (frozen): input_size={spec['input_size']}, feat_dim={model.num_features}")
    return _FROZEN_BACKBONE_CACHE[name]


def build_fresh_backbone(name):
    """Backbone for the per-region encoders (code Group 2, fine-tuned). Builds a
    fresh weight copy independent of the cache, so it never shares parameters
    with the frozen Group 1 backbones."""
    spec = BACKBONE_SPECS[name]
    model = _create_backbone(spec["timm_name"])
    return model, spec["input_size"], model.num_features


def crop_and_resize(img_batch, box, target_size):
    """img_batch: (B,3,224,224) tensor after normalize_image(). box: (x,y,w,h) in pixels."""
    bx, by, bw, bh = box
    crop = img_batch[:, :, by:by+bh, bx:bx+bw]
    return F.interpolate(crop, size=(target_size, target_size), mode='bilinear', align_corners=False)


def crop_and_pad(img_batch, box, target_size, pad_mode="mean"):
    """Same crop, but kept at its NATIVE resolution on a target_size canvas.

    crop_and_resize upscales the south_korea box (48x64 px) to 224x224: 4.7x
    horizontally, 3.5x vertically, and the aspect ratio changes too. A model
    trained on whole 224 scenes then sees cloud structures at a scale it never
    saw in training, so a failure of the whole-image-trained baselines (paper
    Group 2, code group3_*) cannot be attributed to the missing
    context alone. Padding holds the scale fixed and isolates that confound.

    pad_mode:
        "mean"  fill with the crop's own per-image, per-channel mean; neutral,
                no artificial high-frequency edge
        "zero"  fill with 0; the conventional choice, but a 48x64 patch on a
                224 canvas leaves ~94% of the frame black
    """
    bx, by, bw, bh = box
    crop = img_batch[:, :, by:by + bh, bx:bx + bw]
    if bw >= target_size or bh >= target_size:      # nothing to pad (total_patches)
        return F.interpolate(crop, size=(target_size, target_size),
                             mode="bilinear", align_corners=False)

    B, C = crop.shape[0], crop.shape[1]
    if pad_mode == "mean":
        fill = crop.mean(dim=(2, 3), keepdim=True).expand(B, C, target_size, target_size)
        canvas = fill.clone()
    elif pad_mode == "zero":
        canvas = crop.new_zeros((B, C, target_size, target_size))
    else:
        raise ValueError(f"pad_mode must be 'mean' or 'zero', got {pad_mode!r}")

    top = (target_size - bh) // 2
    left = (target_size - bw) // 2
    canvas[:, :, top:top + bh, left:left + bw] = crop
    return canvas
