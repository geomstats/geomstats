import geomstats.backend as gs

if gs.__name__.endswith("pytorch"):
    import torch

    def gpu_is_available():
        return torch.cuda.is_available()

    def to_device(array, device="cuda"):
        # TODO: check autodiff
        return array.to(device)

    def get_device(array):
        return array.device


else:

    def gpu_is_available():
        return False

    def to_device(array, *args, **kwargs):
        return array

    def get_device(array):
        return "cpu"
