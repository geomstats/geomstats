import cv2
import numpy as np


class CovariancDescriptor:
    def __init__(
        self,
        data,
        window=None,
        first_order_kernel=None,
        second_order_kernel=None,
    ):
        self.data = data
        self.window = window
        self.first_order_kernel = first_order_kernel
        self.second_order_kernel = second_order_kernel

    def first_derivative(self, gray_img):
        if kernel is None:
            kernel = self.First_order
        dx = cv2.filter2D(gray_img, cv2.CV_64F, self.first_order_kernel)
        dy = cv2.filter2D(gray_img, cv2.CV_64F, self.first_order_kernel.T)
        return dx, dy

    def second_derivative(self, gray_img):
        if kernel is None:
            kernel = self.second_order
        ddx = cv2.filter2D(gray_img, cv2.CV_64F, self.second_order_kernel)
        ddy = cv2.filter2D(gray_img, cv2.CV_64F, self.second_order_kernel.T)
        return ddx, ddy

    def transform(self):
        dimension = 10
        if center is None:
            center = (image.shape[0] // 2, image.shape[1] // 2)

        rect = [
            center[0] - self.window[0] // 2,
            center[0] + self.window[0] // 2,
            center[1] - self.window[1] // 2,
            center[1] + self.window[1] // 2,
        ]
        image = image[rect[0] - 1 : rect[1] + 1, rect[2] - 1 : rect[3] + 1, :]
        grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        F = np.ones((self.window[0], self.window[1], 10))
        dx, dy = self.first_Derivative(grayimage)
        ddx, ddy = self.second_Derivative(grayimage)
        dx = np.abs(dx)
        dy = np.abs(dy)
        ddx = np.abs(ddx)
        ddy = np.abs(ddy)

        mean = np.sum(F, (0, 1)) / (self.window[0] * self.window[1])
        CovD = np.zeros((dimension, dimension))
        for i in range(self.window[0]):
            for j in range(self.window[1]):
                m = np.mat(F[i, j, :] - mean)
                CovD = CovD + m.T @ m
        return CovD / (self.window[0] * self.window[1] - 1)
