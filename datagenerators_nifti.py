"""
Data loader for custom dataset with .nii.gz format.
Dataset structure:
    data_root/
    ├── labeled/
    │   ├── image/
    │   │   ├── 000_atlas_43_bl.nii.gz  (atlas)
    │   │   └── ...
    │   └── label/
    │       ├── 000_atlas_43_bl.nii.gz
    │       └── ...
    └── unlabeled/
        ├── image/
        │   └── ...  (used for training)
        └── label/
            └── ...  (not used)

Supports both original and preprocessed data (with transform_params.json).
"""

import glob
import os
import numpy as np
import nibabel as nib
import json

# Label range: 0 (background), 1-138 (structures)
NUM_CLASSES = 139  # 0-138 inclusive
ATLAS_FILENAME = '000_atlas_43_bl.nii.gz'


class MRIDatasetNifti(object):
    """Dataset loader for .nii.gz format MRI data."""

    def __init__(self, data_root, num_classes=NUM_CLASSES,
                 batch_size=1, atlas_filename=ATLAS_FILENAME,
                 transform_params=None):
        """
        Args:
            data_root: Root directory containing labeled/ and unlabeled/ folders
            num_classes: Number of classes including background (default: 139)
            batch_size: Batch size for training
            atlas_filename: Name of the atlas file in labeled/image/
            transform_params: Dictionary of transformation parameters (optional)

        Note: Image shape is automatically detected from the first image file.
        """
        self.data_root = data_root
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.atlas_filename = atlas_filename
        self.img_shape = None  # Will be detected automatically
        self.transform_params = transform_params  # For reference

    def load_nifti(self, filepath):
        """Load a .nii.gz file and return as numpy array."""
        img = nib.load(filepath)
        data = img.get_fdata()
        return data

    def load_dataset(self):
        """Load atlas, training (unlabeled), and validation (labeled) data."""
        # Check for transformation parameters (preprocessed data)
        params_path = os.path.join(self.data_root, 'transform_params.json')
        if os.path.exists(params_path):
            print(f"Loading transformation parameters from: {params_path}")
            with open(params_path, 'r') as f:
                self.transform_params = json.load(f)
        else:
            print("No transform_params.json found. Using original data.")
            self.transform_params = None

        labeled_image_dir = os.path.join(self.data_root, 'labeled', 'image')
        labeled_label_dir = os.path.join(self.data_root, 'labeled', 'label')
        unlabeled_image_dir = os.path.join(self.data_root, 'unlabeled', 'image')

        # Load atlas (from labeled folder)
        atlas_path = os.path.join(labeled_image_dir, self.atlas_filename)
        atlas_label_path = os.path.join(labeled_label_dir, self.atlas_filename)

        print(f"Loading atlas: {atlas_path}")
        vol_atlas = self.load_nifti(atlas_path)
        seg_atlas = self.load_nifti(atlas_label_path)

        # Auto-detect image shape from atlas
        self.img_shape = vol_atlas.shape
        print(f"Detected image shape: {self.img_shape}")
        print(f"Atlas label unique: {np.unique(seg_atlas)}")

        # Resize if needed (if atlas shape differs from target)
        vol_atlas = self._normalize(vol_atlas)
        seg_atlas = seg_atlas.astype(np.int32)

        # Load validation data (all labeled data except atlas)
        val_image_files = sorted(glob.glob(os.path.join(labeled_image_dir, '*.nii.gz')))
        val_label_files = sorted(glob.glob(os.path.join(labeled_label_dir, '*.nii.gz')))

        # Remove atlas from validation set
        val_image_files = [f for f in val_image_files if os.path.basename(f) != self.atlas_filename]
        val_label_files = [f for f in val_label_files if os.path.basename(f) != self.atlas_filename]

        print(f"Loading {len(val_image_files)} validation samples...")
        vol_val = []
        seg_val = []
        val_ids = []
        for img_path, seg_path in zip(val_image_files, val_label_files):
            vol = self.load_nifti(img_path)
            seg = self.load_nifti(seg_path)
            vol = self._normalize(vol)
            seg = seg.astype(np.int32)
            vol_val.append(vol)
            seg_val.append(seg)
            val_ids.append(os.path.basename(img_path))

        vol_val = np.array(vol_val, dtype=np.float32)
        seg_val = np.array(seg_val, dtype=np.int32)
        print(f"Validation shape: {vol_val.shape}, unique labels: {np.unique(seg_val)}")

        # Load training data (unlabeled)
        train_image_files = sorted(glob.glob(os.path.join(unlabeled_image_dir, '*.nii.gz')))
        print(f"Loading {len(train_image_files)} training samples...")

        vol_train = []
        train_ids = []
        for img_path in train_image_files:
            vol = self.load_nifti(img_path)
            vol = self._normalize(vol)
            vol_train.append(vol)
            train_ids.append(os.path.basename(img_path))

        vol_train = np.array(vol_train, dtype=np.float32)
        print(f"Training shape: {vol_train.shape}")

        # Add channel dimension: (N, D, H, W) -> (N, D, H, W, 1)
        vol_atlas = vol_atlas[..., np.newaxis]
        seg_atlas = seg_atlas[..., np.newaxis]
        vol_val = vol_val[..., np.newaxis]
        seg_val = seg_val[..., np.newaxis]
        vol_train = vol_train[..., np.newaxis]

        # Convert to channel-first: (N, D, H, W, C) -> (N, C, D, H, W)
        vol_atlas = vol_atlas.transpose(0, 4, 1, 2, 3)
        seg_atlas = seg_atlas.transpose(0, 4, 1, 2, 3)
        vol_val = vol_val.transpose(0, 4, 1, 2, 3)
        seg_val = seg_val.transpose(0, 4, 1, 2, 3)
        vol_train = vol_train.transpose(0, 4, 1, 2, 3)

        self.atlas = (vol_atlas, seg_atlas, [self.atlas_filename])
        self.val = (vol_val, seg_val, val_ids)
        self.train = (vol_train, None, train_ids)  # No labels for training

        print(f"Atlas: {vol_atlas.shape}, Train: {vol_train.shape}, Val: {vol_val.shape}")

        return self.atlas, self.train, self.val

    def _normalize(self, vol):
        """Normalize volume to [0, 1] using 99.99th percentile."""
        max_val = np.percentile(vol, 99.99)
        if max_val > 0:
            vol = np.clip(vol / max_val, 0, 1)
        return vol.astype(np.float32)

    def gen_register_batch(self):
        """Generator for registration training batches."""
        vol_atlas, seg_atlas, _ = self.atlas
        vol_train, _, _ = self.train

        train_num = vol_train.shape[0]
        print(f"Training samples: {train_num}")

        # Tile atlas to batch size
        atlas_batch = np.tile(vol_atlas, (self.batch_size, 1, 1, 1, 1))
        atlas_seg_batch = np.tile(seg_atlas, (self.batch_size, 1, 1, 1, 1))

        while True:
            idx = np.random.choice(train_num, self.batch_size, replace=True)
            train_batch = vol_train[idx]
            yield atlas_batch, atlas_seg_batch, train_batch


def test_data_loading(data_root):
    """Test data loading and print dataset statistics."""
    print("=" * 50)
    print("Testing data loading...")
    print("=" * 50)

    # Initialize dataset (shape will be auto-detected)
    dataset = MRIDatasetNifti(
        data_root=data_root,
        num_classes=139,
        batch_size=1,
        atlas_filename=ATLAS_FILENAME
    )

    # Load dataset
    (vol_atlas, seg_atlas, ids_atlas), \
    (vol_train, _, ids_train), \
    (vol_val, seg_val, ids_val) = dataset.load_dataset()

    print("\n" + "=" * 50)
    print("Dataset Statistics:")
    print("=" * 50)
    print(f"Atlas: volume shape {vol_atlas.shape}, label shape {seg_atlas.shape}")
    print(f"  Atlas label unique values: {np.unique(seg_atlas)}")
    print(f"Training (unlabeled): {vol_train.shape[0]} samples")
    print(f"Validation (labeled): {vol_val.shape[0]} samples")
    print(f"  Validation label unique values: {np.unique(seg_val)}")

    # Check label range
    all_labels = np.unique(seg_val)
    print(f"  Label range: {all_labels.min()} to {all_labels.max()}")
    print(f"  Number of unique labels: {len(all_labels)}")

    print("\n" + "=" * 50)
    print("Data loading test PASSED!")
    print("=" * 50)

    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test data loading')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of the dataset')
    args = parser.parse_args()

    success = test_data_loading(args.data_root)
    if not success:
        exit(1)
