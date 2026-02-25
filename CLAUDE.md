# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a brain MRI segmentation project implementing one-shot neuroanatomy segmentation through online data augmentation and confidence-aware pseudo labeling. The method uses a registration-based approach to augment training data.

## Commands

```bash
# Step 1: Preprocess data (crop zero slices, pad to multiples of 8)
python preprocess.py --data_root /path/to/original/data --output_root /path/to/processed/data

# Step 2: Train the model using processed data
python train_nifti.py --datapath /path/to/processed/data

# Step 3: Test and save predictions
python test_nifti.py --datapath /path/to/processed/data --model_path /path/to/model.pth
```

### Preprocessing Options

- `--data_root`: Root directory of the original dataset
- `--output_root`: Root directory to save processed data
- `--pad_multiple`: Pad dimensions to multiples of this (default: 8)

The preprocessing script will:
1. Crop zero slices from each dimension (where both image and label are all zeros)
2. Pad to make dimensions multiples of 8 (required by the network)
3. Save `transform_params.json` containing transformation parameters for each file

### Training Options

- `--gpu`: GPU device ID (default: '0')
- `--datapath`: Processed data folder path (required)
- `--n_iter`: Number of iterations (default: 10001)
- `--n_save_iter`: Model save frequency (default: 1000)
- `--model_dir`: Output directory for models (default: 'CANDI_Model')

### Testing Options

- `--datapath`: Processed data folder path
- `--model_path`: Path to trained model
- `--output_dir`: Directory to save prediction .nii.gz files (default: 'predictions')

The test script automatically reverses the preprocessing to save predictions in original space.

## Architecture

The codebase uses a custom 3D U-Net style architecture (`network.py`) with registration components:

- **Encoder**: 5-level 3D convolutional encoder with downsampling (filters: 16→32→64→64→128)
- **Decoder**: 4-level upsampling decoder with skip connections, outputs 139 classes (0-138)
- **Reg (Registration)**: Multi-level registration network computing dense displacement fields
- **SpatialTransformer**: Warps volumes using the computed flow field via grid sampling

**Augnet** combines encoder, registration, and decoder modules for the full pipeline.

## Loss Functions (`losses.py`)

- `ncc_loss`: Normalized Cross-Correlation for similarity measurement
- `gradient_loss`: L2 penalty on spatial gradients of flow field (smoothness)
- `entropy_loss`: Entropy-based regularization for segmentation confidence

## Data Format

### Original CANDI Dataset (`datagenerators.py`)

NumPy files in:
- `<datapath>/theone/vol/*_procimg.npy` - Atlas/reference volume
- `<datapath>/train/vol/*_procimg.npy` - Training volumes
- `<datapath>/test/vol/*_procimg.npy` - Test volumes
- Labels: 29 classes with mapping

### Custom NIfTI Dataset (`datagenerators_nifti.py`)

**Dataset structure:**
```
data_root/
├── labeled/
│   ├── image/
│   │   ├── 000_atlas_43_bl.nii.gz  (atlas)
│   │   └── *.nii.gz               (validation images)
│   └── label/
│       ├── 000_atlas_43_bl.nii.gz
│       └── *.nii.gz               (validation labels)
└── unlabeled/
    └── image/
        └── *.nii.gz               (training images, labels NOT used)
```

**Usage:**
- **Atlas**: `000_atlas_43_bl.nii.gz` from labeled folder
- **Validation**: All other files in labeled/ (used for testing)
- **Training**: All files in unlabeled/image/ (labels not used - one-shot setting)

**Labels:** 0 (background), 1-138 (structures) - total 139 classes

**Key differences from original:**
- Uses nibabel to load .nii.gz files
- No label mapping needed (labels are already 1-138)
- Output classes = 139 (was 29)

## Network Output Classes

When using the custom dataset, modify `network.py`:
- **Decoder.out_conv**: Change `nn.Conv3d(filters*2, 29, 1)` to `nn.Conv3d(filters*2, 139, 1)`

## Key Implementation Details

- **Input size requirement**: Network uses 4 levels of strided convolutions and transposed convolutions with stride=2. Input dimensions must be multiples of 8 for proper alignment.
- Training uses registration-based augmentation: warp atlas to training sample, then segment the warped image
- Confidence mask generated via cosine similarity between warped and target features (>0.9 threshold)
- Uses batch normalization by default
- Adam optimizer with learning rate 0.0001
- GPU required (CUDA)

## Dependencies

- PyTorch >= 1.6.0
- NumPy
- nibabel (for .nii.gz medical imaging)
- packaging (for version checks)
- Python >= 3.7
