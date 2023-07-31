import cv2
import numpy as np
import torch
from skimage.filters.rank import mean_bilateral
from skimage import morphology
from PIL import Image
from PIL import ImageEnhance
import argparse

def padCropImg(img):
    H = img.shape[0]
    W = img.shape[1]

    patchRes = 128
    pH = patchRes
    pW = patchRes
    ovlp = int(patchRes * 0.125)  # 32

    padH = (int((H - patchRes) / (patchRes - ovlp) + 1) * (patchRes - ovlp) + patchRes) - H
    padW = (int((W - patchRes) / (patchRes - ovlp) + 1) * (patchRes - ovlp) + patchRes) - W

    padImg = cv2.copyMakeBorder(img, 0, padH, 0, padW, cv2.BORDER_REPLICATE)

    ynum = int((padImg.shape[0] - pH) / (pH - ovlp)) + 1
    xnum = int((padImg.shape[1] - pW) / (pW - ovlp)) + 1

    totalPatch = np.zeros((ynum, xnum, patchRes, patchRes, 3), dtype=np.uint8)

    for j in range(0, ynum):
        for i in range(0, xnum):
            x = int(i * (pW - ovlp))
            y = int(j * (pH - ovlp))
            
            if j == (ynum-1) and i == (xnum-1):
                totalPatch[j, i] = img[-patchRes:, -patchRes:]
            elif j == (ynum-1):
                totalPatch[j, i] = img[-patchRes:, x:int(x + patchRes)]
            elif i == (xnum-1):
                totalPatch[j, i] = img[y:int(y + patchRes), -patchRes:]
            else:
                totalPatch[j, i] = padImg[y:int(y + patchRes), x:int(x + patchRes)]

    return totalPatch, padH, padW


def illCorrection(model, totalPatch):
    totalPatch = totalPatch.astype(np.float32) / 255.0

    ynum = totalPatch.shape[0]
    xnum = totalPatch.shape[1]

    totalResults = np.zeros((ynum, xnum, 128, 128, 3), dtype=np.float32)

    for j in range(0, ynum):
        for i in range(0, xnum):
            patchImg = torch.from_numpy(totalPatch[j, i]).permute(2,0,1)
            patchImg = patchImg.cuda().view(1, 3, 128, 128)

            output = model(patchImg)
            output = output.permute(0, 2, 3, 1).data.cpu().numpy()[0]

            output = output * 255.0
            output = output.astype(np.uint8)

            totalResults[j, i] = output

    return totalResults


def composePatch(totalResults, padH, padW, img):
    ynum = totalResults.shape[0]
    xnum = totalResults.shape[1]
    patchRes = totalResults.shape[2]

    ovlp = int(patchRes * 0.125)
    step = patchRes - ovlp

    resImg = np.zeros((patchRes + (ynum - 1) * step, patchRes + (xnum - 1) * step, 3), np.uint8)
    resImg = np.zeros_like(img).astype('uint8')

    for j in range(0, ynum):
        for i in range(0, xnum):
            sy = int(j * step)
            sx = int(i * step)
            
            if j == 0 and i != (xnum-1):
                resImg[sy:(sy + patchRes), sx:(sx + patchRes)] = totalResults[j, i]
            elif i == 0 and j != (ynum-1):
                resImg[sy+10:(sy + patchRes), sx:(sx + patchRes)] = totalResults[j, i,10:]
            elif j == (ynum-1) and i == (xnum-1):
                resImg[-patchRes+10:, -patchRes+10:] = totalResults[j, i,10:,10:]
            elif j == (ynum-1) and i == 0:
                resImg[-patchRes+10:, sx:(sx + patchRes)] = totalResults[j, i,10:]
            elif j == (ynum-1) and i != 0:
                resImg[-patchRes+10:, sx+10:(sx + patchRes)] = totalResults[j, i,10:,10:]
            elif i == (xnum-1) and j == 0:
                resImg[sy:(sy + patchRes), -patchRes+10:] = totalResults[j, i,:,10:]
            elif i == (xnum-1) and j != 0:
                resImg[sy+10:(sy + patchRes), -patchRes+10:] = totalResults[j, i,10:,10:]
            else:
                resImg[sy+10:(sy + patchRes), sx+10:(sx + patchRes)] = totalResults[j, i,10:,10:]

    resImg[0,:,:] = 255

    return resImg


def preProcess(img):
    img[:,:,0] = mean_bilateral(img[:,:,0], morphology.disk(20), s0=10, s1=10)
    img[:,:,1] = mean_bilateral(img[:,:,1], morphology.disk(20), s0=10, s1=10)
    img[:,:,2] = mean_bilateral(img[:,:,2], morphology.disk(20), s0=10, s1=10)
    
    return img


def postProcess(img):
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Contrast(img)
    factor = 2.0
    img = enhancer.enhance(factor)

    return img


import matplotlib.pyplot as plt
def rec_ill(net, img, saveRecPath):

    totalPatch, padH, padW = padCropImg(img)

    totalResults = illCorrection(net, totalPatch)

    resImg = composePatch(totalResults, padH, padW, img)
    #resImg = postProcess(resImg)
    resImg = Image.fromarray(resImg)
    resImg.save(saveRecPath)  
    # plt.imshow(resImg)
    # plt.show()

import os
from IllTr import IllTr

def reload_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cuda:0')
        print(len(pretrained_dict.keys()))
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model


def rec(opt):
    # print(torch.__version__) # 1.5.1
    img_list = os.listdir(opt.distorrted_path)  # distorted images list
    done_list = os.listdir(opt.isave_path) # done already

    if not os.path.exists(opt.isave_path):  # create save path
        os.mkdir(opt.isave_path)
    
    print("Started rec")
    

    IllTr_model = IllTr().cuda()
    # reload illumination rectification model
    reload_model(IllTr_model, opt.IllTr_path)
    # print("Reload illumination rectification model")

    # To eval mode
    IllTr_model.eval()
    from tqdm import tqdm
    done_list = list(map(lambda x: x.replace('_ill', ''), done_list))
    for img_path in tqdm(img_list):
        if img_path in done_list: continue
        name = img_path.split('.')[-2]  # image name

        # print(opt.distorrted_path, img_path, done_list[:5]); exit()
        
        # img_path = opt.distorrted_path + img_path  # read image and to tensor
        img_path = os.path.join(opt.distorrted_path, img_path)
        img = Image.open(img_path)
        # if len(img.shape) == 2:
        if 1: # Convert grayscale to RGB
            img = img.convert('RGB')
        
        img = np.array(img)
        im_ori = img[:, :, :3] / 255. 
        h, w, _ = im_ori.shape
        im = cv2.resize(im_ori, (288, 288))
        im = im.transpose(2, 0, 1)
        im = torch.from_numpy(im).float().unsqueeze(0)
        
        img_geo = img
        
        with torch.no_grad():
            # # geometric unwarping
            # bm = GeoTr_Seg_model(im.cuda())
            # bm = bm.cpu()
            # bm0 = cv2.resize(bm[0, 0].numpy(), (w, h))  # x flow
            # bm1 = cv2.resize(bm[0, 1].numpy(), (w, h))  # y flow
            # bm0 = cv2.blur(bm0, (3, 3))
            # bm1 = cv2.blur(bm1, (3, 3))
            # lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0)  # h * w * 2
            
            # out = F.grid_sample(torch.from_numpy(im_ori).permute(2,0,1).unsqueeze(0).float(), lbl, align_corners=True)
            # img_geo = ((out[0]*255).permute(1, 2, 0).numpy())[:,:,::-1].astype(np.uint8)
            # cv2.imwrite(opt.gsave_path + name + '_geo' + '.png', img_geo)  # save
            
            # illumination rectification
            if opt.ill_rec:
                ill_savep = opt.isave_path + name + '_ill' + '.png'
                rec_ill(IllTr_model, img_geo, saveRecPath=ill_savep)
        
        print('Done: ', img_path)

def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--distorrted_path',  default='./distorted/')
    # parser.add_argument('--gsave_path',  default='./geo_rec/')
    parser.add_argument('--isave_path',  default='./ill_rec/')
    # parser.add_argument('--Seg_path',  default='./model_pretrained/seg.pth')
    # parser.add_argument('--GeoTr_path',  default='./model_pretrained/geotr.pth')
    parser.add_argument('--IllTr_path',  default='./model_pretrained/illtr.pth')
    parser.add_argument('--ill_rec',  default=True)
    
    opt = parser.parse_args()
    os.makedirs(opt.isave_path, exist_ok=True)

    rec(opt)


if __name__ == '__main__':
    main()
