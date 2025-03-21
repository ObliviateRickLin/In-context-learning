import math

import torch


class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
        "uniform01": Uniform01Sampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        # 如果scale设置为"random_exp"，每次都生成一个新的随机变换矩阵
        if self.scale is not None:
            if isinstance(self.scale, str) and self.scale == "random_exp":
                # 生成一个 n_dims 维的随机对角元素，服从 Exponential(1)
                eigenvalues = torch.empty(self.n_dims).exponential_()
                # 调用 sample_transformation 生成随机变换矩阵，并可以选择是否归一化
                random_scale = sample_transformation(eigenvalues, normalize=True)
                xs_b = xs_b @ random_scale
            else:
                xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b


class Uniform01Sampler(DataSampler):
    """
    Samples uniformly from [0,1]^n_dims.
    """
    def __init__(self, n_dims):
        super().__init__(n_dims)

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        # Sample directly using rand, all in [0,1]
        if seeds is None:
            xs_b = torch.rand(b_size, n_points, self.n_dims)
        else:
            # For reproducibility with seeds
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.rand(n_points, self.n_dims, generator=generator)

        if n_dims_truncated is not None:
            xs_b[:,:,n_dims_truncated:] = 0
        return xs_b
