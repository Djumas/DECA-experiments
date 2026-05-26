# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms
# in the LICENSE file included with this software distribution.
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys
import cv2
import numpy as np
from scipy.io import savemat
import argparse
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def collect_images_recursive(root):
    """Recursively collect image paths under root, skipping result_* output folders."""
    imagepaths = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith('result_')]
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS:
                imagepaths.append(os.path.join(dirpath, filename))
    return sorted(imagepaths)


def savefolder_for_image(imagepath):
    """Output folder next to the source image: result_<basename>."""
    name = os.path.splitext(os.path.basename(imagepath))[0]
    return os.path.join(os.path.dirname(imagepath), f'result_{name}')


def normalize_input_path(path):
    """Accept both / and \\ as separators (common when copying Windows paths into bash)."""
    return os.path.normpath(path.replace('\\', os.sep))


def main(args):
    device = args.device
    inputpath = normalize_input_path(args.inputpath)

    if not os.path.isdir(inputpath):
        print(f'input path does not exist or is not a directory: {inputpath}')
        return

    imagepath_list = collect_images_recursive(inputpath)
    if not imagepath_list:
        print(f'no images found under: {inputpath}')
        return

    print(f'found {len(imagepath_list)} image(s) under {inputpath}')

    testdata = datasets.TestData(imagepath_list, iscrop=args.iscrop, face_detector=args.detector,
                                 sample_step=args.sample_step)

    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = args.extractTex
    deca = DECA(config=deca_cfg, device=device)

    for i in tqdm(range(len(testdata))):
        imagepath = testdata.imagepath_list[i]
        name = testdata[i]['imagename']
        savefolder = savefolder_for_image(imagepath)

        images = testdata[i]['image'].to(device)[None, ...]
        with torch.no_grad():
            codedict = deca.encode(images)
            opdict, visdict = deca.decode(codedict)
            if args.render_orig:
                tform = testdata[i]['tform'][None, ...]
                tform = torch.inverse(tform).transpose(1, 2).to(device)
                original_image = testdata[i]['original_image'][None, ...].to(device)
                _, orig_visdict = deca.decode(codedict, render_orig=True, original_image=original_image, tform=tform)
                orig_visdict['inputs'] = original_image

        if args.saveDepth or args.saveKpt or args.saveObj or args.saveMat or args.saveImages or args.saveRPY or args.saveVis:
            os.makedirs(savefolder, exist_ok=True)

        if args.saveRPY:
            global_pose = codedict['pose'][0, :3].cpu().numpy()
            pitch = global_pose[0] * (180.0 / np.pi)
            yaw = global_pose[1] * (180.0 / np.pi)
            roll = global_pose[2] * (180.0 / np.pi)

            rpy_file_path = os.path.join(savefolder, name + '_rpy.txt')
            with open(rpy_file_path, 'w') as f:
                f.write(f"Pitch (X-axis): {pitch:.4f}\n")
                f.write(f"Yaw   (Y-axis): {yaw:.4f}\n")
                f.write(f"Roll  (Z-axis): {roll:.4f}\n")
                if not testdata[i]['face_detected']:
                    f.write("Face unrecognized\n")

        if args.saveDepth:
            depth_image = deca.render.render_depth(opdict['trans_verts']).repeat(1, 3, 1, 1)
            visdict['depth_images'] = depth_image
            cv2.imwrite(os.path.join(savefolder, name + '_depth.jpg'), util.tensor2image(depth_image[0]))
        if args.saveKpt:
            np.savetxt(os.path.join(savefolder, name + '_kpt2d.txt'), opdict['landmarks2d'][0].cpu().numpy())
            np.savetxt(os.path.join(savefolder, name + '_kpt3d.txt'), opdict['landmarks3d'][0].cpu().numpy())
        if args.saveObj:
            deca.save_obj(os.path.join(savefolder, name + '.obj'), opdict)
        if args.saveMat:
            opdict = util.dict_tensor2npy(opdict)
            savemat(os.path.join(savefolder, name + '.mat'), opdict)
        if args.saveVis:
            cv2.imwrite(os.path.join(savefolder, name + '_vis.jpg'), deca.visualize(visdict))
            if args.render_orig:
                cv2.imwrite(os.path.join(savefolder, name + '_vis_original_size.jpg'), deca.visualize(orig_visdict))
        if args.saveImages:
            for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images',
                             'landmarks2d']:
                if vis_name not in visdict.keys():
                    continue
                cv2.imwrite(os.path.join(savefolder, name + '_' + vis_name + '.jpg'),
                            util.tensor2image(visdict[vis_name][0]))
                if args.render_orig:
                    cv2.imwrite(os.path.join(savefolder, 'orig_' + name + '_' + vis_name + '.jpg'),
                                util.tensor2image(orig_visdict[vis_name][0]))

    print(f'-- processed {len(imagepath_list)} image(s); results saved next to each source file')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DECA reconstruction with RPY angles; recursive input, per-image result folders')

    parser.add_argument('-i', '--inputpath', default='TestSamples/examples', type=str,
                        help='root folder to scan recursively for images (.jpg, .jpeg, .png, .bmp)')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu')
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped')
    parser.add_argument('--sample_step', default=10, type=int,
                        help='sample images from video data for every step')
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details')
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='rasterizer type: pytorch3d or standard')
    parser.add_argument('--render_orig', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to render results in original image size, currently only works when rasterizer_type=standard')
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model')
    parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to extract texture from input image as the uv texture map, set false if you want albeo map from FLAME mode')
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output')
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints')
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image')
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj, detail mesh will end with _detail.obj. \
                            Note that saving objs could be slow')
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat')
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images')
    parser.add_argument('--saveRPY', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save Pitch, Yaw, Roll angles to a text file')

    main(parser.parse_args())
