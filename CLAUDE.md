# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Qwen-Image-Layered is a web application that provides a Gradio-based interface for the Qwen-Image-Layered model, which decomposes images into multiple RGBA layers for editing. The project includes two main web interfaces:

1. **Layer Decomposition** (`src/app.py`) - Decomposes images into RGBA layers and exports to PPTX/ZIP
2. **Layer Editing** (`src/tool/edit_rgba_image.py`) - Edits individual RGBA layers using Qwen-Image-Edit

## System Requirements

- Python with PyTorch
- CUDA-enabled GPU (model runs on CUDA with bfloat16 precision)
- transformers >= 4.51.3 (for Qwen2.5-VL support)
- Latest diffusers library from GitHub

## Dependencies Installation

Install the required dependencies:

```bash
pip install git+https://github.com/huggingface/diffusers
pip install python-pptx
```

Additional dependencies for the editing tool:
- gradio
- torch
- torchvision
- transformers
- PIL (Pillow)

## Running the Application

### Main Decomposition Interface
Launch the layer decomposition web interface on port 7869:

```bash
python src/app.py
```

This starts a Gradio interface at `http://0.0.0.0:7869` where users can:
- Upload images for layer decomposition
- Adjust parameters (seed, guidance scale, inference steps, number of layers)
- Download results as PPTX or ZIP files

### Layer Editing Interface
Launch the RGBA layer editing interface:

```bash
python src/tool/edit_rgba_image.py
```

This provides a separate Gradio interface for editing individual transparent layers using Qwen-Image-Edit-2509.

## Architecture

### src/app.py - Main Decomposition Application

**Core Components:**
- `QwenImageLayeredPipeline` - Pre-trained model loaded from "Qwen/Qwen-Image-Layered"
- `infer()` - Main inference function that processes images and generates layers
- `imagelist_to_pptx()` - Converts layer images to PowerPoint presentation
- Gradio interface with example images from `assets/test_images/`

**Key Parameters:**
- `resolution`: 640 (recommended bucket size; 1024 also available)
- `cfg_normalize`: Whether to enable CFG normalization
- `use_en_prompt`: Automatic caption language (True for EN, False for ZH)
- `layers`: Number of output layers (2-10, default 4)
- `num_inference_steps`: Inference steps (1-50, default 50)
- `true_cfg_scale`: Guidance scale (1.0-10.0, default 4.0)

**Output Format:**
- Gallery of RGBA layer images
- PPTX file with all layers stacked (each layer is independently movable)
- ZIP file containing individual layer PNG files

### src/tool/edit_rgba_image.py - RGBA Layer Editor

**Core Components:**
- `QwenImageEditPlusPipeline` - Model from "Qwen/Qwen-Image-Edit-2509"
- `AutoModelForImageSegmentation` - Background removal using RMBG-2.0
- `infer()` - Applies text-prompted edits to RGBA images
- `blend_with_green_bg()` - Helper to blend transparent images with green background

**Processing Pipeline:**
1. Blends input RGBA image with green background
2. Applies text-prompted edits using diffusion model
3. Removes background using RMBG-2.0 segmentation
4. Returns edited RGBA image with transparency preserved

## Model Behavior Notes

**Important Constraints:**
- The text prompt describes the **overall content** of the input image, including occluded elements
- The prompt does **not** control semantic content of individual layers explicitly
- The model is fine-tuned for image-to-multi-RGBA decomposition
- Text-to-multi-RGBA generation capability is limited

**Decomposition Features:**
- Supports variable-layer decomposition (2-10 layers)
- Supports recursive decomposition (layers can be further decomposed)
- Layers are physically isolated for independent manipulation

## Example Images

Test images are located in `assets/test_images/` (13 example images numbered 1-13).

## Model Loading

Both applications load models on initialization:
- Models are automatically downloaded from HuggingFace on first run
- Models are loaded to CUDA with bfloat16 precision
- Progress bars are enabled for inference tracking
