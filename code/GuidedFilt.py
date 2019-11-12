# -*- coding: utf-8 -*-
"""
Created on Thu May 04 20:09:43 2017
@author: Utkarsh
"""

import cv2
import numpy as np

class GuidedFilter:

    def __init__(self, I, radius, eps):
        #self.I = cv2.CV_32F(I)
        self.I = I
        self.radius = radius
        self.eps = eps

        self.rows = self.I.shape[0]
        self.cols = self.I.shape[1]
        self.chs  = self.I.shape[2]

    def filter(self, p):
        channels = p.shape[2]
        ret = np.zeros_like(p, dtype=np.float32)
        for c in range(channels):
            ret[:, :, c] = self.GuidedFilt(p[:, :, c])
        return ret

    def GuidedFilt(self, p): 
        """
        Parameters
        ----------
        p: NDArray
            Filtering input of 2D
        Returns
        -------
        q: NDArray
            Filtering output of 2D
        """
        # 
        
        p_ = np.expand_dims(p, axis=2)

        meanI = cv2.boxFilter(self.I, -1, (self.radius,  self.radius)) # (H, W, C)
        meanp = cv2.boxFilter(p_, -1, ((2 *self.radius) + 1, (2*self.radius)+1)) # (H, W, 1)
        meanp = np.expand_dims(meanp, axis=2)
        I_ = self.I.reshape((self.rows*self.cols, self.chs, 1)) # (HW, C, 1)
        meanI_ = meanI.reshape((self.rows*self.cols, self.chs, 1)) # (HW, C, 1)

        corrI_ = np.matmul(I_, I_.transpose(0, 2, 1))  # (HW, C, C)
        corrI_ = corrI_.reshape((self.rows, self.cols, self.chs*self.chs)) # (H, W, CC)
        corrI_ = cv2.boxFilter(corrI_, -1, ((2 *self.radius) + 1, (2*self.radius)+1))
        corrI = corrI_.reshape((self.rows*self.cols, self.chs, self.chs)) # (HW, C, C)

        U = np.expand_dims(np.eye(self.chs, dtype=np.float32), axis=0)
        # U = np.tile(U, (self.rows*self.cols, 1, 1)) # (HW, C, C)

        left = np.linalg.inv(corrI + self.eps * U) # (HW, C, C)

        corrIp = cv2.boxFilter(self.I*p_, -1, ((2 *self.radius) + 1, (2*self.radius)+1)) # (H, W, C)
        covIp = corrIp - meanI * meanp # (H, W, C)
        right = covIp.reshape((self.rows*self.cols, self.chs, 1)) # (HW, C, 1)

        a = np.matmul(left, right) # (HW, C, 1)
        axmeanI = np.matmul(a.transpose((0, 2, 1)), meanI_) # (HW, 1, 1)
        axmeanI = axmeanI.reshape((self.rows, self.cols, 1))
        b = meanp - axmeanI # (H, W, 1)
        a = a.reshape((self.rows, self.cols, self.chs))

        meana = cv2.boxFilter(a, -1, ((2 *self.radius) + 1, (2*self.radius)+1))
        meanb = cv2.boxFilter(b, -1, ((2 *self.radius) + 1, (2*self.radius)+1))

        meana = meana.reshape((self.rows*self.cols, 1, self.chs))
        meanb = meanb.reshape((self.rows*self.cols, 1, 1))
        I_ = self.I.reshape((self.rows*self.cols, self.chs, 1))

        q = np.matmul(meana, I_) + meanb
        q = q.reshape((self.rows, self.cols))

        return q
              # eps = 0.015
        # print(r)
        # I = np.double(img)
        # #print("min = {}, max = {}".format(I.min(),I.max()))
        # I2 = cv2.pow(I, 2)
        # mean_I = cv2.boxFilter(I, -1, ((2 * r) + 1, (2 * r) + 1))
        # mean_I2 = cv2.boxFilter(I2, -1, ((2 * r) + 1, (2 * r) + 1))

        # cov_I = mean_I2 - cv2.pow(mean_I, 2)
        # var_I = cov_I

        # a = cv2.divide(cov_I, var_I + eps)
        # b = mean_I - (a * mean_I)

        # mean_a = cv2.boxFilter(a, -1, ((2 * r) + 1, (2 * r) + 1))
        # mean_b = cv2.boxFilter(b, -1, ((2 * r) + 1, (2 * r) + 1))

        # q = (mean_a * I) + mean_b
        # return q