# CLIP → CoreML Exporter (TextEncoder & ImageEncoder)

Exports separate Core ML models for the text and image encoders from a Hugging Face CLIP-like model (e.g., OpenAI CLIP, MobileCLIP variants that expose CLIP*WithProjection classes).

**Based on**: [Queryable PyTorch2CoreML-HuggingFace notebook](https://github.com/mazzzystar/Queryable/blob/main/PyTorch2CoreML-HuggingFace.ipynb)  
**Inspired by**: [Queryable project](https://github.com/mazzzystar/Queryable)

## Features

- ✅ **Robust conversion** with automatic fallback (77→76 tokens) for models that fail validation
- ✅ **Proper token length tracking** to prevent shape mismatches during inference
- ✅ **Separate CoreML models** for text and image encoders for flexible deployment
- ✅ **Validation support** to verify CoreML vs PyTorch output consistency
- ✅ **Image preprocessing options** with embedded normalization support

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Basic Export

```bash
python export_coreml_clip.py \
  --model-id openai/clip-vit-base-patch32 \
  --outdir out \
  --max-seq-len 77
```

### Export with Validation

```bash
python export_coreml_clip.py \
  --model-id openai/clip-vit-base-patch32 \
  --outdir out \
  --max-seq-len 77 \
  --embed-normalize \
  --validate \
  --text "a photo of a cat" \
  --image ./images/cat.jpeg
```

### Test the Exported Models

```bash
python test_cat_inference.py
```

## Command Line Options

- `--model-id`: Hugging Face model ID (e.g., `openai/clip-vit-base-patch32`)
- `--outdir`: Output directory for CoreML models (default: `out`)
- `--max-seq-len`: Maximum sequence length, auto-retries with 76 if 77 fails (default: 77)
- `--deployment-target`: CoreML deployment target (default: `iOS16`, options: `iOS16`, `macOS13`)
- `--embed-normalize`: Embed normalization into the vision model for better accuracy
- `--validate`: Compare CoreML outputs vs PyTorch for validation
- `--text`: Text prompt for validation (default: "a photo of a cat")
- `--image`: Image path for validation

## Output

The script generates two CoreML models:

- **`TextEncoder.mlpackage`**: Converts text prompts to embeddings
  - Input: `prompt` (int32 array, shape: [1, seq_len])
  - Outputs: `embOutput`, `embOutput2` (float32 embeddings)

- **`ImageEncoder.mlpackage`**: Converts images to embeddings  
  - Input: `colorImage` (RGB image, 224×224)
  - Outputs: `embOutput`, `embOutput2` (float32 embeddings)

## Known Issues & Fixes

### Shape Mismatch Error (76 vs 77 tokens)

**Error**: `MultiArray shape (1 x 76) does not match the shape (1 x 77) specified in the model description`

**Solution**: The script now automatically handles this by:
1. Attempting conversion with the requested sequence length (77)
2. If that fails, automatically retrying with 76 tokens
3. Tracking and reporting the actual sequence length used
4. Ensuring validation uses the correct sequence length

Always use the same sequence length for inference that was used during export (logged in the output).

## Testing

The included `test_cat_inference.py` script demonstrates:
- Loading both CoreML models
- Processing an image (automatically resized to 224×224)
- Comparing image embeddings against various text queries
- Ranking similarity scores to identify image content

Example output:
```
🏆 Top matches:
1. 'a photo of a cat' (similarity: 0.2913)
2. 'a cat' (similarity: 0.2795)  
3. 'a cute kitten' (similarity: 0.2647)
```

## Requirements

- Python 3.8+
- PyTorch 2.5.0+ (2.8.0 has warnings but works)
- coremltools
- transformers
- Pillow

See `requirements.txt` for complete dependencies.