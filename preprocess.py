"""
Preprocessing script for MRI data:
1. First pass: analyze all images to find maximum cropped dimensions
2. Crop zero slices from each dimension (where both image and label are all zeros)
3. Pad all images to the same target shape (multiple of 8)
4. Save transformed data and transformation parameters for inverse processing
"""

import os
import glob
import numpy as np
import nibabel as nib
import json
from tqdm import tqdm


def find_crop_bounds(img, label=None, threshold=1e-6):
    """
    Find crop bounds where either image or label has non-zero values.
    """
    img_mask = np.abs(img) > threshold

    if label is not None:
        label_mask = label > 0
        combined_mask = img_mask | label_mask
    else:
        combined_mask = img_mask

    if not np.any(combined_mask):
        return (0, img.shape[0]), (0, img.shape[1]), (0, img.shape[2])

    nonzero_indices = np.array(np.where(combined_mask))
    mins = nonzero_indices.min(axis=1)
    maxs = nonzero_indices.max(axis=1)

    return tuple(zip(mins, maxs + 1))


def get_cropped_shape(img, label=None, threshold=1e-6):
    """Get the cropped shape without actually cropping."""
    crop_bounds = find_crop_bounds(img, label, threshold)
    d = crop_bounds[0][1] - crop_bounds[0][0]
    h = crop_bounds[1][1] - crop_bounds[1][0]
    w = crop_bounds[2][1] - crop_bounds[2][0]
    return (d, h, w)


def crop_and_pad_uniform(img, label=None, target_shape=None, pad_to_multiple=8):
    """
    Crop to non-zero region and pad to uniform target shape.

    Args:
        img: Image array (D, H, W)
        label: Label array (D, H, W), optional
        target_shape: Target shape (D, H, W) - all images will be padded to this
        pad_to_multiple: Pad target_shape to be multiples of this if target_shape is None

    Returns:
        tuple: (transformed_img, transformed_label, params)
    """
    # Find crop bounds
    crop_bounds = find_crop_bounds(img, label)

    # Crop image
    d_start, d_end = crop_bounds[0]
    h_start, h_end = crop_bounds[1]
    w_start, w_end = crop_bounds[2]

    img_cropped = img[d_start:d_end, h_start:h_end, w_start:w_end]

    if label is not None:
        label_cropped = label[d_start:d_end, h_start:h_end, w_start:w_end]
    else:
        label_cropped = None

    # Calculate target shape if not provided
    if target_shape is None:
        d, h, w = img_cropped.shape
        d_target = ((d + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
        h_target = ((h + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
        w_target = ((w + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
        target_shape = (d_target, h_target, w_target)

    # Pad to target shape
    d, h, w = img_cropped.shape
    d_target, h_target, w_target = target_shape

    d_pad_before = 0
    d_pad_after = d_target - d
    h_pad_before = 0
    h_pad_after = h_target - h
    w_pad_before = 0
    w_pad_after = w_target - w

    img_padded = np.pad(
        img_cropped,
        ((d_pad_before, d_pad_after), (h_pad_before, h_pad_after), (w_pad_before, w_pad_after)),
        mode='constant',
        constant_values=0
    )

    if label_cropped is not None:
        label_padded = np.pad(
            label_cropped,
            ((d_pad_before, d_pad_after), (h_pad_before, h_pad_after), (w_pad_before, w_pad_after)),
            mode='constant',
            constant_values=0
        )
    else:
        label_padded = None

    params = {
        'original_shape': img.shape,
        'crop_bounds': {
            'd': [int(d_start), int(d_end)],
            'h': [int(h_start), int(h_end)],
            'w': [int(w_start), int(w_end)]
        },
        'target_shape': list(target_shape),
        'padding': {
            'd': [int(d_pad_before), int(d_pad_after)],
            'h': [int(h_pad_before), int(h_pad_after)],
            'w': [int(w_pad_before), int(w_pad_after)]
        },
        'padded_shape': img_padded.shape
    }

    return img_padded, label_padded, params


def reverse_transform(volume, params):
    """
    Reverse the crop and pad transformations.
    """
    # First, remove padding
    padding = params['padding']
    d_pad_before, d_pad_after = padding['d']
    h_pad_before, h_pad_after = padding['h']
    w_pad_before, w_pad_after = padding['w']

    volume = volume[
        d_pad_before:volume.shape[0] - d_pad_after,
        h_pad_before:volume.shape[1] - h_pad_after,
        w_pad_before:volume.shape[2] - w_pad_after
    ]

    # Then, add back the cropped regions
    crop_bounds = params['crop_bounds']
    d_start, d_end = crop_bounds['d']
    h_start, h_end = crop_bounds['h']
    w_start, w_end = crop_bounds['w']

    original_shape = params['original_shape']

    result = np.zeros(original_shape, dtype=volume.dtype)
    result[d_start:d_end, h_start:h_end, w_start:w_end] = volume

    return result


def process_dataset(data_root, output_root, pad_to_multiple=8):
    """
    Process the entire dataset with uniform target shape.
    """
    labeled_image_dir = os.path.join(data_root, 'labeled', 'image')
    labeled_label_dir = os.path.join(data_root, 'labeled', 'label')
    unlabeled_image_dir = os.path.join(data_root, 'unlabeled', 'image')

    output_labeled_image = os.path.join(output_root, 'labeled', 'image')
    output_labeled_label = os.path.join(output_root, 'labeled', 'label')
    output_unlabeled_image = os.path.join(output_root, 'unlabeled', 'image')

    os.makedirs(output_labeled_image, exist_ok=True)
    os.makedirs(output_labeled_label, exist_ok=True)
    os.makedirs(output_unlabeled_image, exist_ok=True)

    # ============ First pass: find maximum cropped dimensions ============
    print("\n" + "=" * 50)
    print("Step 1: Analyzing dataset to find uniform target shape...")
    print("=" * 50)

    labeled_images = sorted(glob.glob(os.path.join(labeled_image_dir, '*.nii.gz')))
    unlabeled_images = sorted(glob.glob(os.path.join(unlabeled_image_dir, '*.nii.gz')))

    max_d, max_h, max_w = 0, 0, 0

    # Analyze labeled data
    print("Analyzing labeled images...")
    for img_path in tqdm(labeled_images, desc="Labeled"):
        filename = os.path.basename(img_path)
        label_path = os.path.join(labeled_label_dir, filename)

        img = nib.load(img_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        d, h, w = get_cropped_shape(img, label)
        max_d = max(max_d, d)
        max_h = max(max_h, h)
        max_w = max(max_w, w)

    # Analyze unlabeled data
    print("Analyzing unlabeled images...")
    for img_path in tqdm(unlabeled_images, desc="Unlabeled"):
        img = nib.load(img_path).get_fdata()

        d, h, w = get_cropped_shape(img, None)
        max_d = max(max_d, d)
        max_h = max(max_h, h)
        max_w = max(max_w, w)

    # Pad to multiples of 8
    target_d = ((max_d + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
    target_h = ((max_h + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
    target_w = ((max_w + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
    target_shape = (target_d, target_h, target_w)

    print(f"\nMax cropped dimensions: ({max_d}, {max_h}, {max_w})")
    print(f"Target shape (padded to {pad_to_multiple}): {target_shape}")

    # ============ Second pass: process all images ============
    print("\n" + "=" * 50)
    print("Step 2: Processing images to uniform shape...")
    print("=" * 50)

    all_params = {}

    # Process labeled data
    print("\nProcessing labeled data...")
    for img_path in tqdm(labeled_images, desc="Labeled"):
        filename = os.path.basename(img_path)
        label_path = os.path.join(labeled_label_dir, filename)

        img = nib.load(img_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        img_processed, label_processed, params = crop_and_pad_uniform(
            img, label, target_shape, pad_to_multiple
        )

        img_nifti = nib.Nifti1Image(img_processed, np.eye(4))
        label_nifti = nib.Nifti1Image(label_processed.astype(np.int32), np.eye(4))

        nib.save(img_nifti, os.path.join(output_labeled_image, filename))
        nib.save(label_nifti, os.path.join(output_labeled_label, filename))

        all_params[filename] = params

    # Process unlabeled data
    print("\nProcessing unlabeled data...")
    for img_path in tqdm(unlabeled_images, desc="Unlabeled"):
        filename = os.path.basename(img_path)

        img = nib.load(img_path).get_fdata()

        img_processed, _, params = crop_and_pad_uniform(img, None, target_shape, pad_to_multiple)

        img_nifti = nib.Nifti1Image(img_processed, np.eye(4))
        nib.save(img_nifti, os.path.join(output_unlabeled_image, filename))

        all_params[filename] = params

    # Save transformation parameters
    params_path = os.path.join(output_root, 'transform_params.json')
    with open(params_path, 'w') as f:
        json.dump(all_params, f, indent=2)

    # Save target shape for reference
    metadata = {
        'target_shape': list(target_shape),
        'pad_to_multiple': pad_to_multiple,
        'max_cropped_dims': [max_d, max_h, max_w]
    }
    metadata_path = os.path.join(output_root, 'preprocess_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 50}")
    print("Processing complete!")
    print(f"{'=' * 50}")
    print(f"Output saved to: {output_root}")
    print(f"Parameters saved to: {params_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"\nAll processed images have uniform shape: {target_shape}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess MRI data')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of the original dataset')
    parser.add_argument('--output_root', type=str, required=True,
                        help='Root directory to save processed data')
    parser.add_argument('--pad_multiple', type=int, default=8,
                        help='Pad dimensions to multiples of this (default: 8)')

    args = parser.parse_args()

    process_dataset(args.data_root, args.output_root, args.pad_multiple)
