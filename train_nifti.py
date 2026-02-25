"""
Training script for custom NIfTI dataset.
Uses one-shot learning with online data augmentation.
"""

import os
import sys
import random
from argparse import ArgumentParser

import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F
from losses import ncc_loss, gradient_loss, entropy_loss
import logging
from network import Augnet, SpatialTransformer
from datagenerators_nifti import MRIDatasetNifti, ATLAS_FILENAME


def train(args):
    """Train the model on unlabeled data with atlas supervision."""
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    logging.basicConfig(
        filename=os.path.join(args.model_dir, 'log.log'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # Load dataset
    print(f"Loading dataset from: {args.datapath}")
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

    train_gen = dataset.gen_register_batch()

    # Initialize model
    model = Augnet(img_shape, normalization='batchnorm')
    trf = SpatialTransformer(img_shape, 'nearest')
    trf2 = SpatialTransformer(img_shape)
    model.train()
    model.cuda()

    opt = Adam(model.parameters(), lr=0.0001)

    for i in range(0, args.n_iter):
        # Save model checkpoint
        if i % args.n_save_iter == 0:
            save_file_name = os.path.join(args.model_dir, '%d.pth' % i)
            torch.save(model.state_dict(), save_file_name)
            logging.info(f"Saved model at iteration {i}")

        # Generate batches
        atlas_batch, atlas_seg_batch, train_batch = next(train_gen)

        atlas_batch = torch.from_numpy(atlas_batch).cuda()
        train_batch = torch.from_numpy(train_batch).cuda()
        atlas_seg_batch = torch.from_numpy(atlas_seg_batch).cuda()

        # Forward pass
        atlas_feats, train_feats = model.encoder(atlas_batch), model.encoder(train_batch)
        flow = model.reg(atlas_feats, train_feats)
        train_out = model.decoder(train_feats)
        warp_batch = model.reg.spa1(atlas_batch, flow)

        # Confidence-aware pseudo labeling
        aug_flow = flow.detach()
        warp_batch2 = trf2(atlas_batch, aug_flow).detach()
        warp_seg = trf(atlas_seg_batch.float(), aug_flow)
        warp_f = trf2(atlas_feats[0].detach(), aug_flow)
        sim = F.cosine_similarity(warp_f, train_feats[0].detach(), dim=1)
        mask = (sim > 0.9).float()

        # Segmentation loss on warped data
        out = model.decoder(model.encoder(warp_batch2))
        seg_loss = F.cross_entropy(out, warp_seg.long().squeeze(1))
        seg_loss2 = F.cross_entropy(train_out, warp_seg.long().squeeze(1), reduction='none')
        seg_loss2 = (seg_loss2 * mask).sum() / (mask.sum() + 1e-6)

        # Registration losses
        sim_loss = ncc_loss(warp_batch, train_batch)
        grad_loss = gradient_loss(flow)

        # Total loss
        loss = sim_loss + grad_loss + seg_loss + 0.1 * seg_loss2

        logging.info(f"{i},{loss.item():.6f},{sim_loss.item():.6f},"
                     f"{grad_loss.item():.6f},{seg_loss.item():.6f},{seg_loss2.item():.6f}")

        opt.zero_grad()
        loss.backward()
        opt.step()

    # Save final model
    save_file_name = os.path.join(args.model_dir, 'final_model.pth')
    torch.save(model.state_dict(), save_file_name)
    logging.info(f"Training complete. Final model saved to {save_file_name}")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--gpu", type=str, default='0',
                        help="GPU device ID")
    parser.add_argument("--datapath", type=str, required=True,
                        help="Root directory of the dataset")
    parser.add_argument("--n_iter", type=int, default=10001,
                        help="Number of iterations")
    parser.add_argument("--n_save_iter", type=int, default=1000,
                        help="Frequency of model saves")
    parser.add_argument("--model_dir", type=str, default='ADNI_Model',
                        help="Output directory for models")

    args = parser.parse_args()

    train(args)
