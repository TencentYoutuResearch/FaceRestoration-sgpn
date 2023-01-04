import os
import argparse
import cv2
import numpy as np
import torch
import utils.cropface as cropface
from utils.umeyama import umeyama
from face_detection.yolov5_face import YoloFace
from face_d3dfr.preprocess import D3DFR_preprocess_class
from face_d3dfr.D3DFR_ReconModel import model_FR3D_wrapper, model_3DMM_wrapper
from face_restoration.model_sgpn import FullGenerator_SGPN


# FaceRestoration
class FaceRestoration(object):
    def __init__(self, args):
        # face det
        self.yoloface = YoloFace(pt_path=args.ckpt_facedet, device=args.device)
        # 3dmm
        self.model_3dmm = model_3DMM_wrapper(args.ckpt_3dmm, device=args.device)
        # preprocess
        self.d3dfrpreprocess = D3DFR_preprocess_class(args.ckpt_3dmm)
        # d3dfr
        self.d3dfr = model_FR3D_wrapper().eval().to(args.device)
        info = self.d3dfr.model_FR3d.load_state_dict(
            torch.load(args.ckpt_fr3d, map_location=lambda storage, loc: storage))
        print('load d3dfr: ', info)
        self.sgpn = FullGenerator_SGPN().eval().to(args.device)
        info = self.sgpn.load_state_dict(torch.load(args.ckpt_sgpn, map_location=lambda storage, loc: storage))
        print('load sgpn: ', info)

        self.in_size = args.in_size
        self.device = args.device

        # the mask for pasting restored faces back
        self.mask = np.zeros((args.in_size, args.in_size), np.float32)
        cv2.rectangle(self.mask, (26, 26), (args.in_size-26, args.in_size-26), (1, 1, 1), -1, cv2.LINE_AA)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 4)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 4)
        self.mask = torch.from_numpy(self.mask).to(args.device).view(1, 1, args.in_size, args.in_size)

        # get the reference 5 landmarks position in the crop settings
        default_square = True
        inner_padding_factor = 0.25
        outer_padding = (0, 0)
        self.reference_5pts = cropface.get_reference_facial_points((args.in_size, args.in_size), inner_padding_factor,
                                                                   outer_padding, default_square)

    @torch.no_grad()
    def infer(self, img_crop_cv2, crop_pts5):
        # convert tensor
        crop_rgb = cv2.cvtColor(img_crop_cv2, cv2.COLOR_BGR2RGB)
        crop_rgb_normed = (torch.from_numpy(crop_rgb).permute(2, 0, 1).float() - 127.5) / 127.5
        crop_t = crop_rgb_normed.view(1, 3, self.in_size, self.in_size).to(self.device)

        # warp
        d3dfr_5pts = self.d3dfrpreprocess.get_D3DFR_target5p(crop_pts5)
        tfm_d3dfr = umeyama(src=crop_pts5, dst=d3dfr_5pts, estimate_scale=True)[0:2]
        tfm_inv_d3dfr = umeyama(src=d3dfr_5pts, dst=crop_pts5, estimate_scale=True)[0:2]
        tfm_d3dfr_tensor = torch.tensor(np.array(tfm_d3dfr).astype(np.float32)).unsqueeze(0).to(self.device)
        tfm_inv_d3dfr_tensor = torch.tensor(np.array(tfm_inv_d3dfr).astype(np.float32)).unsqueeze(0).to(self.device)

        lr_coeff = self.d3dfr(im=crop_t, warpmat=tfm_d3dfr_tensor)
        recon_result_dict = self.model_3dmm(D3D_coeff=lr_coeff, inverse_warpmat=tfm_inv_d3dfr_tensor)
        crop_3drec_t = recon_result_dict['im_3DRec']

        # inference
        result_t = self.sgpn(im_LR=crop_t, im_D3D=crop_3drec_t, coeff_D3D=lr_coeff).clamp(-1, 1)
        return crop_t, crop_3drec_t, result_t


    @torch.no_grad()
    def enhance(self, img_cv2, real_image=True):
        if real_image:
            h, w, c = img_cv2.shape
            _, kpss, _ = self.yoloface.detect(img_cv2)
            face_num = len(kpss)

            # enhance faces one by one
            crop_list = []
            for i in range(face_num):
                pts5 = kpss[i]
                padones = np.ones((5, 1), dtype=pts5.dtype)
                tfm = umeyama(src=pts5, dst=self.reference_5pts, estimate_scale=True)[0:2]
                crop_pts5 = np.concatenate((pts5, padones), axis=1).dot(tfm.T)

                crop = cv2.warpAffine(img_cv2, tfm, (self.in_size, self.in_size), flags=cv2.INTER_CUBIC)
                crop_t, crop_3drec_t, result_t = self.infer(crop, crop_pts5)
                result_t = result_t*self.mask+crop_t*(1-self.mask)

                crop_en_bgr = cv2.cvtColor(
                    (result_t[0] * 127.5 + 127.5).cpu().numpy().astype('uint8').transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                crop_list.append(crop_en_bgr)
                # warp back
                img_cv2 = cv2.warpAffine(crop_en_bgr, tfm, (w, h), dst=img_cv2,
                                         flags=(cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP),
                                         borderMode=cv2.BORDER_TRANSPARENT)
            return img_cv2
        else:
            # used for cropped images
            return self.enhance_crop(img_cv2)

    @torch.no_grad()
    def enhance_crop(self, img_crop_cv2):
        img_crop_cv2 = cv2.resize(img_crop_cv2, (self.in_size, self.in_size), cv2.INTER_CUBIC)
        _, kpss, _ = self.yoloface.detect(img_crop_cv2)

        crop_pts5 = kpss[0]
        _, crop_3drec_t, result_t = self.infer(img_crop_cv2, crop_pts5)
        crop_3d_bgr = cv2.cvtColor((crop_3drec_t[0] * 127.5 + 127.5).cpu().numpy().astype('uint8').transpose(1, 2, 0),
                                   cv2.COLOR_RGB2BGR)
        crop_en_bgr = cv2.cvtColor((result_t[0] * 127.5 + 127.5).cpu().numpy().astype('uint8').transpose(1, 2, 0),
                                   cv2.COLOR_RGB2BGR)
        return cv2.hconcat([crop_3d_bgr, crop_en_bgr])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SGPN-512', help='SGPN model')
    parser.add_argument('--in_size', type=int, default=512, help='in resolution of SGPN')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='inference device')
    parser.add_argument('--ckpt_facedet', type=str, default='checkpoints/yoloface_v5m.pt', help='face detector')
    parser.add_argument('--ckpt_3dmm', type=str, default='checkpoints/BFM/', help='3dmm')
    parser.add_argument('--ckpt_fr3d', type=str, default='checkpoints/d3dfr_finetune_ours.pth', help='d3dfr')
    parser.add_argument('--ckpt_sgpn', type=str, default='checkpoints/sgpn.pth', help='sgpn generator')
    parser.add_argument('--real_image', action='store_true', help='if realimage, warp back to origin image')
    parser.add_argument('--src_path', type=str, default='examples/test/', help='input folder')
    parser.add_argument('--res_path', type=str, default='examples/restoration/', help='output folder')
    parser.add_argument('--format', type=str, default='.jpg', help='extension of output')
    args = parser.parse_args()
    print(args)

    # init
    face_restoration_inst = FaceRestoration(args)

    for im_file in os.listdir(args.src_path):
        if im_file.endswith(args.format):
            print(im_file)
            im_cv2 = cv2.imread(os.path.join(args.src_path, im_file))
            if not args.real_image:
                im_cv2 = cv2.resize(im_cv2, (args.in_size, args.in_size), cv2.INTER_CUBIC)

            res = face_restoration_inst.enhance(im_cv2.copy(), args.real_image)
            # save image
            cv2.imwrite(os.path.join(args.res_path, im_file), cv2.hconcat([im_cv2, res]))
