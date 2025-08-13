
### `export_coreml_clip.py`

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import coremltools as ct
from PIL import Image

from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizerFast,
    CLIPVisionModelWithProjection,
    CLIPProcessor,
)

import torchvision.transforms as T


def _log(msg):
    print(f"[export] {msg}")


# ---------------------------
# Text Encoder (export)
# ---------------------------
class TextWrapper(torch.nn.Module):
    """
    Wraps HF CLIPTextModelWithProjection to accept int32 tokens.
    Most HF models expect torch.long (int64); we upcast inside forward.
    Returns the same tuple ordering as the HF model with return_dict=False.
    """
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids: torch.Tensor):
        return self.m(input_ids.to(torch.int64))


def export_text_encoder(model_id: str, outdir: Path, max_seq_len: int, deployment_target: str):
    _log(f"Loading text model: {model_id}")
    text_model = CLIPTextModelWithProjection.from_pretrained(model_id, return_dict=False)
    tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
    text_model.eval()

    def _try_convert(seq_len: int, out_path: Path):
        _log(f"Tracing text model with seq_len={seq_len}")
        ex = tokenizer(
            "a photo of a cat",
            return_tensors="pt",
            padding="max_length",
            max_length=seq_len,
            truncation=True,
        )["input_ids"].to(torch.int32)

        traced = torch.jit.trace(TextWrapper(text_model), ex)

        _log(f"Converting text model to Core ML ({deployment_target})")
        mlmodel = ct.convert(
            traced,
            convert_to="mlprogram",
            minimum_deployment_target=getattr(ct.target, deployment_target),
            inputs=[ct.TensorType(name="prompt", shape=[1, seq_len], dtype=np.int32)],
            outputs=[
                ct.TensorType(name="embOutput", dtype=np.float32),
                ct.TensorType(name="embOutput2", dtype=np.float32),
            ],
        )
        mlmodel.save(str(out_path))
        return out_path

    out_path = outdir / "TextEncoder.mlpackage"
    # Known quirk: some combos fail validation at 77. Auto-retry at 76.
    actual_seq_len = max_seq_len
    try:
        return _try_convert(max_seq_len, out_path), actual_seq_len
    except Exception as e:
        if max_seq_len == 77:
            _log(f"Conversion failed with seq_len=77 ({e}). Retrying with 76…")
            actual_seq_len = 76
            return _try_convert(76, out_path), actual_seq_len
        raise


# ---------------------------
# Image Encoder (export)
# ---------------------------
class VisionWrapper(torch.nn.Module):
    """
    Optionally embeds Normalize(mean, std) *inside* the traced model to minimize Core ML preprocessing drift.
    If embed_normalize=True, the model expects inputs in [0,1] (channel-first).
    If False, the model expects already-normalized tensors (HF processor-style).
    """
    def __init__(self, m, mean, std, embed_normalize: bool):
        super().__init__()
        self.m = m
        self.embed_normalize = embed_normalize
        self.norm = T.Normalize(mean=mean, std=std)

    def forward(self, x):
        # x: float32 tensor [N,3,H,W]
        if self.embed_normalize:
            x = self.norm(x)
        return self.m(x)


def _get_hw_from_processor(proc) -> tuple[int, int]:
    # Robustly pull image size from the processor
    # HF processors vary: size could be dict with height/width or shortest_edge, or crop_size.
    ip = proc.image_processor
    H = W = 224
    if hasattr(ip, "size") and isinstance(ip.size, dict):
        if "height" in ip.size and "width" in ip.size:
            H, W = ip.size["height"], ip.size["width"]
        elif "shortest_edge" in ip.size:
            H = W = ip.size["shortest_edge"]
    elif hasattr(ip, "crop_size") and isinstance(ip.crop_size, dict):
        H, W = ip.crop_size.get("height", H), ip.crop_size.get("width", W)
    return int(H), int(W)


def export_image_encoder(
    model_id: str,
    outdir: Path,
    deployment_target: str,
    embed_normalize: bool,
):
    _log(f"Loading vision model: {model_id}")
    vision_model = CLIPVisionModelWithProjection.from_pretrained(model_id, return_dict=False)
    processor = CLIPProcessor.from_pretrained(model_id)
    vision_model.eval()

    # Shapes & normalization stats
    mean = processor.image_processor.image_mean
    std = processor.image_processor.image_std
    H, W = _get_hw_from_processor(processor)

    # Build example inputs
    # We always get a "normalized" example from the processor to learn the shape.
    dummy_img = Image.new("RGB", (W, H), (128, 128, 128))
    processed = processor(images=dummy_img, return_tensors="pt")
    norm_example = processed["pixel_values"]  # [1,3,H,W], normalized

    if embed_normalize:
        # traced model expects unnormalized [0,1]; use a random tensor in [0,1] with the same shape
        example_input = torch.rand_like(norm_example)
    else:
        # traced model expects normalized tensor (same as HF processor produces)
        example_input = norm_example

    traced = torch.jit.trace(VisionWrapper(vision_model, mean, std, embed_normalize), example_input)

    # Core ML input type
    if embed_normalize:
        # Let Core ML only scale to [0,1] from uint8 image; normalization happens inside the traced graph
        image_input = ct.ImageType(
            name="colorImage",
            color_layout=ct.colorlayout.RGB,
            shape=example_input.shape,  # [1,3,H,W]
            scale=1.0 / 255.0,
            bias=[0.0, 0.0, 0.0],
            channel_first=True,
        )
    else:
        # Match HF Normalize via scale/bias approx (single scale). This can introduce tiny drift.
        # (This mirrors the notebook approach.)
        bias = [-mean[i] / std[i] for i in range(3)]
        scale = 1.0 / (std[0] * 255.0)
        image_input = ct.ImageType(
            name="colorImage",
            color_layout=ct.colorlayout.RGB,
            shape=example_input.shape,
            scale=scale,
            bias=bias,
            channel_first=True,
        )

    _log(f"Converting image model to Core ML ({deployment_target})")
    img_mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=getattr(ct.target, deployment_target),
        inputs=[image_input],
        outputs=[
            ct.TensorType(name="embOutput", dtype=np.float32),
            ct.TensorType(name="embOutput2", dtype=np.float32),
        ],
    )
    out_path = outdir / "ImageEncoder.mlpackage"
    img_mlmodel.save(str(out_path))
    return out_path, processor, traced, embed_normalize


# ---------------------------
# Validation (optional)
# ---------------------------
def _validate_text(text_ml_path: Path, traced_text, tokenizer, text: str, seq_len: int):
    try:
        ml = ct.models.MLModel(str(text_ml_path))
    except Exception as e:
        _log(f"Skipping text validation (cannot load MLModel): {e}")
        return

    toks = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=seq_len)["input_ids"]
    toks32 = toks.to(torch.int32)
    pt_out = traced_text(toks32)
    ml_out = ml.predict({"prompt": toks32.numpy()})

    a = pt_out[0].detach().cpu().numpy()[0]
    b = ml_out["embOutput"][0]
    l2 = np.linalg.norm(a - b)
    _log(f"[validate/text] L2 diff embOutput vs PyTorch: {l2:.6f}")


def _validate_image(img_ml_path: Path, traced_vision, processor, image_path: str | None, embed_normalize: bool):
    try:
        ml = ct.models.MLModel(str(img_ml_path))
    except Exception as e:
        _log(f"Skipping image validation (cannot load MLModel): {e}")
        return

    if image_path and Path(image_path).exists():
        pil = Image.open(image_path).convert("RGB")
    else:
        H, W = _get_hw_from_processor(processor)
        pil = Image.new("RGB", (W, H), (200, 120, 60))

    # Core ML prediction
    ml_pred = ml.predict({"colorImage": pil})
    ml_emb = ml_pred["embOutput"][0]

    # PyTorch prediction (match traced input expectations)
    if embed_normalize:
        # traced model expects [0,1], channel-first
        # Start from HF-normalized tensor and undo normalization back to [0,1]
        t = processor(images=pil, return_tensors="pt")["pixel_values"]  # normalized
        mean = torch.tensor(processor.image_processor.image_mean).view(1, 3, 1, 1)
        std = torch.tensor(processor.image_processor.image_std).view(1, 3, 1, 1)
        unnormalized = t * std + mean  # back to [0,1]
        pt_out = traced_vision(unnormalized)
    else:
        # traced expects HF-normalized directly
        t = processor(images=pil, return_tensors="pt")["pixel_values"]
        pt_out = traced_vision(t)

    a = pt_out[0].detach().cpu().numpy()[0]
    l2 = np.linalg.norm(a - ml_emb)
    _log(f"[validate/image] L2 diff embOutput vs PyTorch: {l2:.6f}")


# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Export CLIP-like encoders to Core ML (separate text/image).")
    parser.add_argument("--model-id", required=True, help="HF model id, e.g. openai/clip-vit-base-patch32")
    parser.add_argument("--outdir", default="out", help="Output directory")
    parser.add_argument("--max-seq-len", type=int, default=77, help="Max sequence length (auto-retries 76 if 77 fails)")
    parser.add_argument("--deployment-target", default="iOS16", choices=["iOS16", "macOS13"], help="Core ML target")
    parser.add_argument("--embed-normalize", action="store_true", help="Embed Normalize(mean,std) into vision graph")
    parser.add_argument("--validate", action="store_true", help="Compare MLModel vs PyTorch for given text/image")
    parser.add_argument("--text", default="a photo of a cat", help="Text prompt for validation")
    parser.add_argument("--image", default=None, help="Path to an RGB image for validation")

    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # TEXT
    try:
        text_ml_path, actual_seq_len = export_text_encoder(
            model_id=args.model_id,
            outdir=outdir,
            max_seq_len=args.max_seq_len,
            deployment_target=args.deployment_target,
        )
        _log(f"Saved: {text_ml_path} (using seq_len={actual_seq_len})")
    except Exception as e:
        _log(f"Text encoder export failed: {e}")
        sys.exit(1)

    # Load again for validation (same tokenizer + traced wrapper)
    text_model = CLIPTextModelWithProjection.from_pretrained(args.model_id, return_dict=False)
    tokenizer = CLIPTokenizerFast.from_pretrained(args.model_id)
    text_model.eval()
    ex = tokenizer(
        "a photo of a cat",
        return_tensors="pt",
        padding="max_length",
        max_length=actual_seq_len,  # use the actual seq_len that was used for conversion
        truncation=True,
    )["input_ids"].to(torch.int32)
    traced_text = torch.jit.trace(TextWrapper(text_model), ex)

    # IMAGE
    try:
        image_ml_path, processor, traced_vision, embed_norm = export_image_encoder(
            model_id=args.model_id,
            outdir=outdir,
            deployment_target=args.deployment_target,
            embed_normalize=args.embed_normalize,
        )
        _log(f"Saved: {image_ml_path}")
    except Exception as e:
        _log(f"Image encoder export failed: {e}")
        sys.exit(1)

    # VALIDATION (optional)
    if args.validate:
        # Use the actual seq_len that was used for conversion
        _validate_text(text_ml_path, traced_text, tokenizer, args.text, actual_seq_len)
        _validate_image(image_ml_path, traced_vision, processor, args.image, embed_norm)

    _log("Done.")


if __name__ == "__main__":
    main()
