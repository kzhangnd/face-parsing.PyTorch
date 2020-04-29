#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet
import sys

import torch

import os
from os import path, listdir, makedirs
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm


def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride,
                                  fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] + '_mask.png', vis_parsing_anno)
        cv2.imwrite(save_path[:-4] + '_color_mask.png', vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # return vis_im


def evaluate(source='./data', dest='./res/test_res', cp='evaluate.pth'):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if path.isfile(source):
        full_path = True
        dspth = np.sort(np.loadtxt(source, dtype=np.str))
    else:
        full_path = False
        dspth = listdir(source)

    with torch.no_grad():
        for image_name in tqdm(dspth):
            if not full_path:
                image_path = path.join(source, image_name)
            else:
                image_path = image_name
                image_name = path.split(image_name)[1]

            img = Image.open(image_path)

            if full_path:
                sub_folder = path.basename(path.normpath(path.split(image_path)[0]))

                dest_path = path.join(dest, sub_folder)

                if not path.exists(dest_path):
                    makedirs(dest_path)

            else:
                dest_path = dest

            save_path = osp.join(dest_path, image_name)

            if os.path.isfile(save_path[:-4] + '_mask.png'):
                print('Skipping...')
                continue

            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(parsing)
            # print(np.unique(parsing))

            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=save_path)


if __name__ == "__main__":
    source = sys.argv[1]
    dest = sys.argv[2]
    evaluate(source=source, dest=dest, cp='79999_iter.pth')
