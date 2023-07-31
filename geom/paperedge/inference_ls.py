# -*- encoding: utf-8 -*-
import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from networks.paperedge import GlobalWarper, LocalWarper, WarperUtil

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def load_img(img_path):
    im = cv2.imread(img_path).astype(np.float32) / 255.0
    im = im[:, :, (2, 1, 0)]
    im = cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA)
    im = torch.from_numpy(np.transpose(im, (2, 0, 1)))
    return im


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Enet_ckpt', type=str,
                        default='models/G_w_checkpoint_13820.pt')
    parser.add_argument('--Tnet_ckpt', type=str,
                        default='models/L_w_checkpoint_27640.pt')
    # parser.add_argument('--img_path', type=str, default='images/1.jpg')
    parser.add_argument('--distorted_path',  default='./../distorted/')

    parser.add_argument('--out_dir', type=str, default='output')
    args = parser.parse_args()

    import glob
    img_list = glob.glob(args.distorted_path + '/*')  # distorted images list
    from copy import deepcopy
    
    # for img_path in img_list:
    if 1:
        # print(img_path)

        # img_path = args.img_path
        dst_dir = args.out_dir
        Path(dst_dir).mkdir(parents=True, exist_ok=True)

        

        netG = GlobalWarper().to('cuda')
        netG.load_state_dict(torch.load(args.Enet_ckpt)['G'])
        netG.eval()

        netL = LocalWarper().to('cuda')
        netL.load_state_dict(torch.load(args.Tnet_ckpt)['L'])
        netL.eval()

        warpUtil = WarperUtil(64).to('cuda')
    
    import time
    durations = []
    img_names = []
    img_list = list(filter(lambda x: not x.endswith('.csv'), img_list))

    for img_path in img_list:

        im = cv2.imread(img_path).astype(np.float32) / 255.0
        start = time.time()
        im = torch.from_numpy(np.transpose(im, (2, 0, 1)))
        im = im.to('cuda').unsqueeze(0)

        from pathlib import Path
        img_name = Path(img_path).name

        gs_d, ls_d = None, None
        with torch.no_grad():
            x = load_img(img_path)
            x = x.unsqueeze(0)
            x = x.to('cuda')
            # d = netG(x)  # d_E the edged-based deformation field
            # d = warpUtil.global_post_warp(d, 64)
            # gs_d = copy.deepcopy(d)

            # d = F.interpolate(d, size=256, mode='bilinear', align_corners=True)
            # y0 = F.grid_sample(x, d.permute(0, 2, 3, 1), align_corners=True)

            y0 = x.clone()

            nit = 2
            # nit = 5
            # y = y0.clone()


            ls_in = y0
            ls_y_prev = im
            for it in range(nit):
                # print('ls_in', ls_in.shape)
                ls_d = netL(ls_in) # [1, 256, 256, 3] => [1, 2, 64, 64]
                # print('ls_d (model output)', ls_d.shape)
                ls_d = F.interpolate(ls_d, size=256, mode='bilinear', align_corners=True) # [1, 2, 64, 64] => [1, 2, 256, 256]
                # print('ls_d (after interpolation)', ls_d.shape)
                ls_d = ls_d.clamp(-1.0, 1.0) # [1, 2, 256, 256]

                # gs_y = ls_in.clone()
                # gs_y = deepcopy(im)
                gs_y = ls_y_prev

                # print('gs_y', gs_y.shape)
                ls_d = F.interpolate(ls_d, (gs_y.size(2), gs_y.size(3)), mode='bilinear', align_corners=True)
                # gs_y shape: [1, 3, 256, 256]
                # print('ls_d (after interpolation with gs)', ls_d.shape)
                ls_y = F.grid_sample(gs_y, ls_d.permute(0, 2, 3, 1), align_corners=True).detach() # [1, 3, 1442, 1075])
                # print('ls_y (grid_sample)', ls_y.shape)
                ls_y_prev = deepcopy(ls_y)
                ls_y = ls_y.squeeze().permute(1, 2, 0).cpu().numpy() # original image
                # print('ls_y after squeeze permute)', ls_y.shape)
                # ls_in = ls_y.clone()

                # resize ls_y
                ls_in = deepcopy(ls_y)
                ls_in = cv2.resize(ls_in, (256, 256), interpolation=cv2.INTER_AREA)
                # ls_in = torch.from_numpy(ls_y).unsqueeze(0).cuda()
                ls_in = torch.from_numpy(np.transpose(ls_in, (2, 0, 1))).unsqueeze(0).cuda()
                # import ipdb; ipdb.set_trace()

        

        # gs_d = F.interpolate(gs_d, (im.size(2), im.size(3)), mode='bilinear', align_corners=True)
        # gs_y = F.grid_sample(im, gs_d.permute(0, 2, 3, 1), align_corners=True).detach()
        # tmp_y = gs_y.squeeze().permute(1, 2, 0).cpu().numpy()
        # # cv2.imwrite(f'{dst_dir}/result_gs.png', tmp_y * 255.)
        # cv2.imwrite(f'{dst_dir}/{img_name}_gs.png', tmp_y * 255.)
        
        # cv2.imwrite(f'{dst_dir}/result_ls.png', ls_y * 255.)
        # ls_y = torch.from_numpy(ls_y)
        # print("********", ls_y.shape)
        # ls_y = ls_y.squeeze().permute(1, 2, 0).cpu().numpy()
        # print(ls_y.shape)
        dur = time.time() - start
        print(dur)
        durations.append(dur)
        cv2.imwrite(f'{dst_dir}/{img_name}', ls_y * 255.)
        img_names.append(img_name)
        # break
    
    from pathlib import Path
    import pandas as pd
    import os

    pd.DataFrame(
        {'img': img_names, 'distortion_runtime': durations}
    ).to_csv(os.path.join(dst_dir, 'metadata.csv'))
