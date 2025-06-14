#!/usr/bin/env python3
"""
End-to-end script for segmenting cells with Cellpose, extracting single-cell images,
classifying them using a ViT model, drawing boxes for apoptosis and pyroptosis,
and saving predictions to CSV and images to output directory.
Assumes 'best_weight_VitTrans.pth' is located in the same directory as this script by default.
"""
import os
import sys
import argparse
import subprocess
import csv

import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import timm
from cellpose import models as cpmodels


def run_cellpose_api(input_dir, use_gpu):
    """
    Segment images in input_dir using Cellpose Python API,
    and save masks as <basename>_seg.npy files.
    """
    cpmodel = cpmodels.Cellpose(gpu=use_gpu, model_type='cyto')
    for fn in os.listdir(input_dir):
        if not fn.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
            continue
        img_path = os.path.join(input_dir, fn)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read {img_path}, skipping.")
            continue
        masks, flows, styles, diams = cpmodel.eval(
            img, channels=[0, 0], diameter=None
        )
        out_path = os.path.join(input_dir, fn.rsplit('.', 1)[0] + '_seg.npy')
        np.save(out_path, {'masks': masks})
        print(f"Saved segmentation: {out_path}")


def load_model(model_path, device):
    """
    Load a ViT model with pretrained weights for 3 classes.
    """
    NUM_CLASSES = 3
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=NUM_CLASSES)
    checkpoint = torch.load(model_path, map_location='cpu')
    # If checkpoint contains a 'model' key, extract it
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        checkpoint = checkpoint['model']
    # Remove any incompatible head parameters
    for key in list(checkpoint.keys()):
        if key.startswith('head'):
            del checkpoint[key]
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Segment cells, classify with ViT, draw boxes, and save results.'
    )
    parser.add_argument(
        '--input_dir', '-i', required=True,
        help='Directory containing images to process'
    )
    parser.add_argument(
        '--output_dir', '-o', default='boxed',
        help='Name of subdirectory under input_dir to save boxed images'
    )
    parser.add_argument(
        '--csv_path', '-c', default='results.csv',
        help='Path to save the CSV with predictions'
    )
    parser.add_argument(
        '--model_path', '-m', default='best_weight_VitTrans.pth',
        help="Path to the ViT model weights .pth file (default: best_weight_VitTrans.pth)"
    )
    parser.add_argument(
        '--use_gpu', action='store_true',
        help='Enable GPU for Cellpose and PyTorch inference'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help='Confidence threshold for drawing boxes (default: 0.5)'
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = os.path.join(input_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Prepare CSV file
    with open(args.csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'filename', 'cell_id', 'x1', 'y1', 'x2', 'y2',
            'pred_class', 'class_name', 'confidence'
        ])

    # Determine device for PyTorch
    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    if args.use_gpu and device.type == 'cpu':
        print('Warning: GPU requested but not available, using CPU instead.', file=sys.stderr)

    # 1. Run Cellpose segmentation
    run_cellpose_api(input_dir, args.use_gpu)

    # 2. Load ViT model
    model = load_model(args.model_path, device)

    # 3. Prepare transforms
    normalize = T.Normalize(mean=[0.5115]*3, std=[0.1316]*3)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        normalize
    ])

    # Class mapping
    CLASS_MAP = {0: 'apoptosis', 1: 'pyroptosis', 2: 'other'}

    # 4. Process each segmentation file
    for fname in os.listdir(input_dir):
        if not fname.endswith('_seg.npy'):
            continue
        basename = fname[:-8]  
        seg_path = os.path.join(input_dir, fname)

        # Find original image
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
            candidate = os.path.join(input_dir, basename + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            print(f"No image found for {basename}, skipping.", file=sys.stderr)
            continue

        # Load segmentation masks
        seg_data = np.load(seg_path, allow_pickle=True).item()
        masks = seg_data.get('masks', None)
        if masks is None:
            print(f"No 'masks' key in {seg_path}, skipping.", file=sys.stderr)
            continue

        # Load image
        img = Image.open(img_path).convert('RGB')
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Collect patches and bboxes
        patches, bboxes, cell_ids = [], [], []
        for label in np.unique(masks):
            if label == 0:
                continue
            ys, xs = np.where(masks == label)
            if ys.size == 0:
                continue
            y1, y2 = ys.min(), ys.max()
            x1, x2 = xs.min(), xs.max()
            pad = 5
            y1, y2 = max(y1 - pad, 0), min(y2 + pad, masks.shape[0])
            x1, x2 = max(x1 - pad, 0), min(x2 + pad, masks.shape[1])
            patch = img.crop((x1, y1, x2, y2))
            patches.append(transform(patch))
            bboxes.append((x1, y1, x2, y2))
            cell_ids.append(label)

        if not patches:
            continue

        # Batch inference
        batch = torch.stack(patches).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        confs = probs[np.arange(len(preds)), preds]

        with open(args.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for cid, (x1, y1, x2, y2), pred, conf in zip(cell_ids, bboxes, preds, confs):
                cls_name = CLASS_MAP.get(int(pred), str(pred))
                writer.writerow([
                    basename, cid, x1, y1, x2, y2,
                    int(pred), cls_name, f"{conf:.4f}"
                ])
                if pred in [0, 1] and conf >= args.threshold:
                    color = (0, 255, 0) if pred == 0 else (0, 0, 255)
                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)

        out_path = os.path.join(output_dir, f"{basename}_boxed.png")
        cv2.imwrite(out_path, img_cv)
        print(f"Processed {basename}, saved to {out_path}")

    print("All images processed.")

if __name__ == '__main__':
    main()
