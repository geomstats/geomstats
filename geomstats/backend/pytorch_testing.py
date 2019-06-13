"""Pytorch based testing backend."""

import torch


def assert_allclose(*args, **kwargs):
    return torch.testing.assert_allclose(*args, **kwargs)
