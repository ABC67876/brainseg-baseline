# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a brain MRI segmentation project implementing one-shot neuroanatomy segmentation through online data augmentation and confidence-aware pseudo labeling. The method uses a registration-based approach to augment training data.

## Commands

```bash
# Train the model (default: 10001 iterations)
python train.py

# Test the model
python test.py
```

### Training Options

- `--gpu`: GPU device ID (default: '0')
- `--datapath`: Data folder path (default: './CANDI')
- `--n_iter`: Number of iterations (default: 10001)
- `--n_save_iter`: Model save frequency (default: 1000)
- `--model_dir`: Output directory for models (default: 'CANDI_Model')

### Testing Options

The test script loads the model from `./CANDI_Model/10000.pth` by default and saves Dice scores to `res.npy`.

## Architecture

The codebase uses a custom 3D U-Net style architecture (`network.py`) with registration components:

- **Encoder**: 5-level 3D convolutional encoder with downsampling (filters: 16→32→64→64→128)
- **Decoder**: 4-level upsampling decoder with skip connections, outputs 29 classes
- **Reg (Registration)**: Multi-level registration network computing dense displacement fields
- **SpatialTransformer**: Warps volumes using the computed flow field via grid sampling

**Augnet** combines encoder, registration, and decoder modules for the full pipeline.

## Loss Functions (`losses.py`)

- `ncc_loss`: Normalized Cross-Correlation for similarity measurement
- `gradient_loss`: L2 penalty on spatial gradients of flow field (smoothness)
- `entropy_loss`: Entropy-based regularization for segmentation confidence

## Data Format (`datagenerators.py`)

Data is stored as NumPy files in:
- `<datapath>/theone/vol/*_procimg.npy` - Atlas/reference volume
- `<datapath>/train/vol/*_procimg.npy` - Training volumes
- `<datapath>/test/vol/*_procimg.npy` - Test volumes
- Corresponding segmentations in `seg/` subdirectories

Volume shape: (160, 160, 128) with a single channel dimension
Labels: 29 classes (brain regions) with label mapping defined in code

## Key Implementation Details

- Training uses registration-based augmentation: warp atlas to training sample, then segment the warped image
- Confidence mask generated via cosine similarity between warped and target features (>0.9 threshold)
- Uses batch normalization by default
- Adam optimizer with learning rate 0.0001
- GPU required (CUDA)

## Dependencies

- PyTorch >= 1.6.0
- NumPy
- nibabel (for medical imaging)
- packaging (for version checks)
- Python >= 3.7
