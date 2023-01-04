import os
import cv2
import numpy as np
import torch
from utils.umeyama import umeyama
from face_detection.yolov5_face import YoloFace
from face_d3dfr.preprocess import D3DFR_preprocess_class
from face_d3dfr.D3DFR_ReconModel import model_FR3D_wrapper, model_3DMM_wrapper

# test_device
test_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# init face det
yoloface = YoloFace(pt_path='checkpoints/yoloface_v5m.pt')
# init d3dfrpreprocess
d3dfrpreprocess = D3DFR_preprocess_class('checkpoints/BFM/')
# model_3dmm
model_3dmm = model_3DMM_wrapper('checkpoints/BFM/')

# origin
model_fr3d_orign = model_FR3D_wrapper().to(test_device)
ckpt1 = os.path.join('checkpoints/', 'd3dfr_opensource.pth')
info = model_fr3d_orign.model_FR3d.load_state_dict(torch.load(ckpt1, map_location=lambda storage, loc: storage))
print('load model_fr3d_orign: ', info)

# finetuned
model_fr3d_finetune = model_FR3D_wrapper().to(test_device)
ckpt2 = os.path.join('checkpoints/', 'd3dfr_finetune_ours.pth')
info = model_fr3d_finetune.model_FR3d.load_state_dict(torch.load(ckpt2, map_location=lambda storage, loc: storage))
print('load model_fr3d_finetune ', info)

# test image
imgpath = 'examples/test_crop/test.jpg'

srcimg = cv2.imread(imgpath)
srcimg = cv2.resize(srcimg, (512, 512))
h, w, c = srcimg.shape
bboxes, kpss, scores = yoloface.detect(srcimg)

srcimg_copy = srcimg.copy()
if len(bboxes) > 0:
    for i in range(bboxes.shape[0]):
        xmin, ymin, xamx, ymax = int(bboxes[i, 0]), int(bboxes[i, 1]), int(bboxes[i, 0] + bboxes[i, 2]), int(
            bboxes[i, 1] + bboxes[i, 3])
        cv2.rectangle(srcimg_copy, (xmin, ymin), (xamx, ymax), (0, 0, 255), thickness=2)
        for j in range(5):
            cv2.circle(srcimg_copy, (int(kpss[i, j, 0]), int(kpss[i, j, 1])), 1, (0, 255, 0), thickness=5)
        cv2.putText(srcimg_copy, str(round(scores[i], 3)), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    thickness=1)

    pts5 = kpss[0]
    tgt_5p = d3dfrpreprocess.get_D3DFR_target5p(pts5)
    warpmat_D3DFR = umeyama(src=pts5, dst=tgt_5p, estimate_scale=True)[0:2]
    inverse_warpmat_D3DFR = umeyama(src=tgt_5p, dst=pts5, estimate_scale=True)[0:2]
    warpmat_D3DFR_tensor = torch.tensor(np.array(warpmat_D3DFR).astype(np.float32)).unsqueeze(0).to(test_device)
    inverse_warpmat_D3DFR_tensor = torch.tensor(np.array(inverse_warpmat_D3DFR).astype(np.float32)).unsqueeze(0).to(
        test_device)

    srcimg_rgb = cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB)
    srcimg_rgb_normed = (torch.from_numpy(srcimg_rgb).permute(2, 0, 1).float() - 127.5) / 127.5
    im_t = srcimg_rgb_normed.view(1, 3, h, w).to(test_device)
    origin_coeff = model_fr3d_orign(im=im_t, warpmat=warpmat_D3DFR_tensor)
    recon_result_dict = model_3dmm(D3D_coeff=origin_coeff, inverse_warpmat=inverse_warpmat_D3DFR_tensor)
    origin_3DRec_t = recon_result_dict['im_3DRec']
    finetune_coeff = model_fr3d_finetune(im=im_t, warpmat=warpmat_D3DFR_tensor)
    recon_result_dict = model_3dmm(D3D_coeff=finetune_coeff, inverse_warpmat=inverse_warpmat_D3DFR_tensor)
    finetune_3DRec_t = recon_result_dict['im_3DRec']

    # print(im_3DRec_t.size())
    # print(im_3DRec_t)

    origin_recon_bgr = cv2.cvtColor(
        (origin_3DRec_t[0] * 127.5 + 127.5).cpu().numpy().astype('uint8').transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
    finetune_recon_bgr = cv2.cvtColor(
        (finetune_3DRec_t[0] * 127.5 + 127.5).cpu().numpy().astype('uint8').transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
    # print(recon_bgr.shape)
    # save image
    cv2.imwrite('examples/test_d3dfr.jpg', cv2.hconcat([srcimg, srcimg_copy, origin_recon_bgr, finetune_recon_bgr]))
