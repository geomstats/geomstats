import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as Tf

from geomstats.geometry.spd_matrices import SPDMatrices


class CovarianceDescriptor(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel_x = (
            1 / 4 * torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        )
        self.second_x = (
            1
            / 32
            * torch.tensor(
                [
                    [1.0, 0.0, -2.0, 0.0, 1.0],
                    [4.0, 0.0, -8.0, 0.0, 4.0],
                    [6.0, 0.0, -12.0, 0.0, 6.0],
                    [4.0, 0.0, -8.0, 0.0, 4.0],
                    [1.0, 0.0, -2.0, 0.0, 1.0],
                ]
            )
        )

    def derivative_features(self, img):
        if img.shape[1] == 3:
            img = T.Grayscale()(img)
        smoothed_img = Tf.gaussian_blur(img, kernel_size=(3, 3), sigma=(0.2, 0.2))
        first_filters = torch.stack([self.sobel_x, self.sobel_x.T], axis=0).unsqueeze(1)
        second_filters = torch.stack(
            [self.second_x, self.second_x.T], axis=0
        ).unsqueeze(1)
        first_abs_derivatives = torch.abs(F.conv2d(smoothed_img, first_filters))
        second_abs_derivtaives = torch.abs(F.conv2d(smoothed_img, second_filters))
        norm_first_derivatives = torch.sqrt(
            first_abs_derivatives[:, 0, :, :] ** 2
            + first_abs_derivatives[:, 1, :, :] ** 2
        ).unsqueeze(1)
        angle = torch.atan2(
            first_abs_derivatives[:, 0, :, :], first_abs_derivatives[:, 1, :, :]
        ).unsqueeze(1)
        return [
            first_abs_derivatives[:, :, 1:-1, 1:-1],
            second_abs_derivtaives,
            norm_first_derivatives[:, :, 1:-1, 1:-1],
            angle[:, :, 1:-1, 1:-1],
        ]

    def other_features(self, img):
        h = img.shape[-1]
        grid_x, grid_y = torch.meshgrid(
            torch.arange(img.shape[-1]), torch.arange(img.shape[-1])
        )
        broadcasted_grid_x = (
            torch.broadcast_to(
                grid_x, (img.shape[0], 1, img.shape[-1], img.shape[-2])
            ).float()
            / h
        )
        broadcasted_grid_y = (
            torch.broadcast_to(
                grid_y, (img.shape[0], 1, img.shape[-1], img.shape[-2])
            ).float()
            / h
        )
        return [
            broadcasted_grid_x[:, :, 2:-2, 2:-2],
            broadcasted_grid_y[:, :, 2:-2, 2:-2],
            img[:, :, 2:-2, 2:-2],
        ]

    def covar(self, features, noise=1e-6):

        assert len(features.shape) == 3
        _, D, N = features.shape

        centered = features - torch.sum(features, 2, keepdim=True) / N
        cov = torch.einsum("ijk,ilk->ijl", centered, centered) / (N)
        return cov + noise * torch.eye(D)

    def forward(self, img):
        assert len(img.shape) == 4
        derivative_features = self.derivative_features(img)
        other_features = self.other_features(img)
        features = torch.cat(other_features + derivative_features, axis=1)
        vectorized = features.reshape(features.shape[0], features.shape[1], -1)
        return self.covar(vectorized)
