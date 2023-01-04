import os
import numpy as np
from scipy.io import loadmat
from utils.umeyama import umeyama

# default crop size
D3DFR_DEFAULT_CROP_SIZE = (224, 224)


# calculating least square problem for image alignment
def POS(xp, x):
    npts = xp.shape[1]

    A = np.zeros([2 * npts, 8])

    A[0:2 * npts - 1:2, 0:3] = x.transpose()
    A[0:2 * npts - 1:2, 3] = 1

    A[1:2 * npts:2, 4:7] = x.transpose()
    A[1:2 * npts:2, 7] = 1

    b = np.reshape(xp.transpose(), [2 * npts, 1])

    k, _, _, _ = np.linalg.lstsq(A, b)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2
    t = np.stack([sTx, sTy], axis=0)

    return t, s


# D3DFR_preprocess_class
class D3DFR_preprocess_class():
    def __init__(self, checkpoint_path, orig_img_size=512):
        self.FILE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.lm3D = self.load_lm3d(checkpoint_path)
        self.img_size = orig_img_size

    def load_lm3d(self, checkpoint_path):
        Lm3D = loadmat(os.path.join(checkpoint_path, 'similarity_Lm3D_all.mat'))
        Lm3D = Lm3D['lm']
        # calculate 5 facial landmarks using 68 landmarks
        lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
        Lm3D = np.stack([Lm3D[lm_idx[0], :], np.mean(Lm3D[lm_idx[[1, 2]], :], 0), np.mean(
            Lm3D[lm_idx[[3, 4]], :], 0), Lm3D[lm_idx[5], :], Lm3D[lm_idx[6], :]], axis=0)
        Lm3D = Lm3D[[1, 2, 0, 3, 4], :]
        return Lm3D

    # get affine trans mat
    def get_warp_mat(self, lm5p):
        tgt_5p = self.get_D3DFR_target5p(lm5p)
        warp_mat = umeyama(src=lm5p, dst=tgt_5p, estimate_scale=True)[0:2]
        return warp_mat

    # get 5 points
    def get_D3DFR_target5p(self, lm5p):
        lm = lm5p.copy()
        lm[:, -1] = self.img_size - 1 - lm[:, -1]
        target_lm5p = self.align_img(lm)
        target_lm5p[:, -1] = 224 - 1 - target_lm5p[:, -1]
        return target_lm5p

    # utils for face reconstruction
    def align_img(self, lm5p,  target_size=224., rescale_factor=102.):

        # calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
        t, s = POS(lm5p.transpose(), self.lm3D.transpose())
        s = rescale_factor / s

        # processing the image
        lm_new = self.resize_n_crop_img(lm5p, t, s, target_size=target_size)

        return lm_new

    # resize_n_crop_img
    def resize_n_crop_img(self, lm, t, s, target_size=224.):
        w0, h0 = self.img_size, self.img_size
        w = (w0 * s).astype(np.int32)
        h = (h0 * s).astype(np.int32)

        lm = np.stack([lm[:, 0] - t[0] + w0 / 2, lm[:, 1] - t[1] + h0 / 2], axis=1) * s
        lm = lm - np.reshape(np.array([(w / 2 - target_size / 2), (h / 2 - target_size / 2)]), [1, 2])
        return lm
