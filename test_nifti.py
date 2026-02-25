"""
Test script for custom NIfTI dataset with preprocessing support.
Evaluates the trained model on validation data and saves predictions.
"""

import os
import sys
import argparse
import logging
import json

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from network import Augnet, SpatialTransformer
from datagenerators_nifti import MRIDatasetNifti, ATLAS_FILENAME
from preprocess import reverse_transform


def dice(vol1, vol2, labels=None, nargout=1):
    """
    Dice volume overlap metric.
    Excludes background (label 0) by default.
    """
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))

    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        top = 2 * np.sum(np.logical_and(vol1 == lab, vol2 == lab))
        bottom = np.sum(vol1 == lab) + np.sum(vol2 == lab)
        bottom = np.maximum(bottom, np.finfo(float).eps)
        dicem[idx] = top / bottom

    if nargout == 1:
        return dicem
    else:
        return (dicem, labels)


def test(args):
    """Run inference on validation set, compute Dice, and save predictions."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # Load transformation parameters
    params_path = os.path.join(args.datapath, 'transform_params.json')
    if os.path.exists(params_path):
        print(f"Loading transformation parameters from: {params_path}")
        with open(params_path, 'r') as f:
            transform_params = json.load(f)
    else:
        print(f"WARNING: No transformation parameters found at {params_path}")
        print("Predictions will be saved in processed space (cropped and padded).")
        transform_params = None

    # Load dataset
    print(f"\nLoading dataset from: {args.datapath}")
    dataset = MRIDatasetNifti(
        data_root=args.datapath,
        num_classes=139,
        batch_size=1,
        atlas_filename=ATLAS_FILENAME
    )

    (vol_atlas, seg_atlas, ids_atlas), \
    (vol_train, _, ids_train), \
    (vol_val, seg_val, ids_val) = dataset.load_dataset()

    # Get shape from dataset
    img_shape = dataset.img_shape

    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model = Augnet(img_shape)
    model.load_state_dict(torch.load(args.model_path))
    model.cuda()
    model.eval()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Compute Dice scores for validation set
    num_classes = 138  # 1-138
    dice_scores = np.zeros((vol_val.shape[0], num_classes))

    for i in range(vol_val.shape[0]):
        filename = ids_val[i]
        vol = vol_val[i:i+1]
        seg_true = seg_val[i:i+1]

        # Transpose: (N, C, D, H, W) -> (N, D, H, W, C)
        vol = vol.transpose(0, 4, 1, 2, 3)
        seg_true = seg_true.transpose(0, 4, 1, 2, 3)

        vol = torch.from_numpy(vol).cuda()

        with torch.no_grad():
            out = model.decoder(model.encoder(vol))
            pre = out.argmax(1, keepdim=True)

        pre = pre.cpu().numpy()
        seg_true = seg_true.squeeze()  # Remove batch dim
        pre_squeezed = pre.squeeze()

        # Reverse transformation if params available
        if transform_params is not None and filename in transform_params:
            params = transform_params[filename]
            # Reverse: first reverse the network output, then apply transform
            # The pre is already in processed space (cropped and padded)
            pre_reversed = reverse_transform(pre_squeezed, params)
        else:
            pre_reversed = pre_squeezed

        # Save prediction in original space (same format as label folder)
        pred_nifti = nib.Nifti1Image(pre_reversed.astype(np.int32), np.eye(4))
        output_path = os.path.join(args.output_dir, filename)
        nib.save(pred_nifti, output_path)

        # Compute Dice for labels 1-138
        # Use the reversed prediction for Dice calculation
        if transform_params is not None and filename in transform_params:
            # Also reverse ground truth for fair comparison
            params = transform_params[filename]
            seg_true_reversed = reverse_transform(seg_true, params)
            dic = dice(pre_reversed, seg_true_reversed, labels=list(range(1, 139)))
        else:
            dic = dice(pre_squeezed, seg_true, labels=list(range(1, 139)))

        # Handle case where some labels might be missing
        if len(dic) < num_classes:
            dic_full = np.zeros(num_classes)
            dic_full[:len(dic)] = dic
            dic = dic_full

        dice_scores[i] = dic
        print(f"{filename}: Mean Dice = {dic.mean():.4f}")

    # Print final results
    print("\n" + "=" * 50)
    print("Final Results:")
    print("=" * 50)
    mean_per_case = dice_scores.mean(axis=1)
    print(f"Mean Dice (per case): {mean_per_case.mean():.4f} ± {mean_per_case.std():.4f}")
    print(f"Min: {mean_per_case.min():.4f}, Max: {mean_per_case.max():.4f}")
    print(f"Overall Mean Dice: {dice_scores.mean():.4f}")

    # Save results
    np.save(args.output, mean_per_case)
    print(f"\nResults saved to: {args.output}")
    print(f"Predictions saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test on custom NIfTI dataset')

    parser.add_argument("--gpu", type=str, default='0',
                        help="GPU device ID")
    parser.add_argument("--datapath", type=str, required=True,
                        help="Root directory of the processed dataset")
    parser.add_argument("--model_path", type=str, default='./CANDI_Model/10000.pth',
                        help="Path to trained model")
    parser.add_argument("--output", type=str, default='res.npy',
                        help="Output file for Dice scores")
    parser.add_argument("--output_dir", type=str, default='predictions',
                        help="Directory to save prediction nifti files")

    args = parser.parse_args()

    test(args)
