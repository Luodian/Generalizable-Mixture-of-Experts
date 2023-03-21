# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO
import glob

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
from collections import OrderedDict

sys.path.append("/mnt/lustre/bli/projects/EIL")
from domainbed import vision_transformer, vision_transformer_hybrid


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask, (10, 10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str,
                        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_path", default="/mnt/lustre/bli/data/domain_net/real", type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(224, 224), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='./attn_output_vit', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build & load model
    model_path = '{project_path}/sweep/output/{exp_name}/d2c8a444c1472737722e9354afe0f994/model.pkl'
    model = vision_transformer.deit_small_patch16_224(pretrained=True, num_classes=0, moe_interval=24, num_experts=4, Hierachical=False).cuda()
    state_dict = torch.load(model_path)['model_dict']
    only_weights = OrderedDict()
    for item in state_dict.keys():
        if 'head' not in item:
            only_weights[item.replace('model.', '')] = state_dict[item]

    for p in model.parameters():
        p.requires_grad = False

    # model.load_state_dict(only_weights, strict=False)
    model.eval()
    import pickle

    image_list = []

    # for filename in glob.glob("/mnt/lustre/bli/data/domain_net/real/**/*.jpg"):
    #     image_list.append(filename)
    #
    # random.shuffle(image_list)
    # image_list = image_list[:1000]
    #
    # with open('test_image_list.pkl', 'wb') as fp:
    #     pickle.dump(image_list, fp)

    with open('test_image_list.pkl', 'rb') as fp:
        image_list = pickle.load(fp)

    for img_full_path in image_list:
        img_name = img_full_path.split('/')[-1]
        if img_full_path is None:
            # user has not specified any image - we use our own image
            print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
            print("Since no image path have been provided, we take the first image in our paper.")
            response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
            img = Image.open(BytesIO(response.content))
            img = img.convert('RGB')
        elif os.path.isfile(img_full_path):
            with open(img_full_path, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
        else:
            print(f"Provided image path {img_full_path} is non valid.")
            sys.exit(1)
        transform = pth_transforms.Compose([
            pth_transforms.Resize(args.image_size),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        img = transform(img)

        # make the image divisible by the patch size
        w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
        img = img[:, :w, :h].unsqueeze(0)

        w_featmap = img.shape[-2] // args.patch_size
        h_featmap = img.shape[-1] // args.patch_size

        attentions = model.get_last_selfattention(img.to(device))

        nh = attentions.shape[1]  # number of head

        # we keep only the output patch attention
        attentions = attentions[0, :, 0, 2:].reshape(nh, -1)

        if args.threshold is not None:
            # we keep only a certain percentage of the mass
            val, idx = torch.sort(attentions)
            val /= torch.sum(val, dim=1, keepdim=True)
            cumval = torch.cumsum(val, dim=1)
            th_attn = cumval > (1 - args.threshold)
            idx2 = torch.argsort(idx)
            for head in range(nh):
                th_attn[head] = th_attn[head][idx2[head]]
            th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
            # interpolate
            th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

        # save attentions heatmaps
        os.makedirs(args.output_dir, exist_ok=True)
        torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(args.output_dir, img_name))
        for j in range(nh):
            fname = os.path.join(args.output_dir, "{}_attn_head".format(img_name.replace('.jpg', '')) + str(j) + ".png")
            plt.imsave(fname=fname, arr=attentions[j], format='png')
            print(f"{fname} saved.")

        if args.threshold is not None:
            image = skimage.io.imread(img_full_path)
            from skimage.transform import rescale, resize, downscale_local_mean

            image = resize(image, (224, 224), anti_aliasing=True)
            for j in range(nh):
                display_instances(image, th_attn[j], fname=os.path.join(args.output_dir, "{}_mask_th".format(img_name.replace('.jpg', '')) + str(args.threshold) + "_head" + str(j) + ".png"), blur=False)
