import math

import torch


def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()


def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()


def cross_entropy(ys_pred, ys):
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)


class Task:
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(
    task_name, n_dims, batch_size, pool_dict=None, num_tasks=None, **kwargs
):
    task_names_to_classes = {
        "linear_regression": LinearRegression,
        "sparse_linear_regression": SparseLinearRegression,
        "linear_classification": LinearClassification,
        "noisy_linear_regression": NoisyLinearRegression,
        "quadratic_regression": QuadraticRegression,
        "relu_2nn_regression": Relu2nnRegression,
        "decision_tree": DecisionTree,
        "gaussian_kernel_regression": GaussianKernelRegression,
        "example1": Example1Task,
        "example2": Example2Task,
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError


class LinearRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.w_b = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class SparseLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        sparsity=3,
        valid_coords=None,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(SparseLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.sparsity = sparsity
        if valid_coords is None:
            valid_coords = n_dims
        assert valid_coords <= n_dims

        for i, w in enumerate(self.w_b):
            mask = torch.ones(n_dims).bool()
            if seeds is None:
                perm = torch.randperm(valid_coords)
            else:
                generator = torch.Generator()
                generator.manual_seed(seeds[i])
                perm = torch.randperm(valid_coords, generator=generator)
            mask[perm[:sparsity]] = False
            w[mask] = 0

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class LinearClassification(LinearRegression):
    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        return ys_b.sign()

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy


class NoisyLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        noise_std=0,
        renormalize_ys=False,
    ):
        """noise_std: standard deviation of noise added to the prediction."""
        super(NoisyLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.noise_std = noise_std
        self.renormalize_ys = renormalize_ys

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        ys_b_noisy = ys_b + torch.randn_like(ys_b) * self.noise_std
        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()

        return ys_b_noisy


class QuadraticRegression(LinearRegression):
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b_quad = ((xs_b**2) @ w_b)[:, :, 0]
        #         ys_b_quad = ys_b_quad * math.sqrt(self.n_dims) / ys_b_quad.std()
        # Renormalize to Linear Regression Scale
        # The function is y = scale * ((x^2 * w) / 3)
        ys_b_quad = ys_b_quad / math.sqrt(3)
        ys_b_quad = self.scale * ys_b_quad
        return ys_b_quad


class Relu2nnRegression(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        hidden_layer_size=100,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(Relu2nnRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.hidden_layer_size = hidden_layer_size

        if pool_dict is None and seeds is None:
            self.W1 = torch.randn(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.randn(self.b_size, hidden_layer_size, 1)
        elif seeds is not None:
            self.W1 = torch.zeros(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.zeros(self.b_size, hidden_layer_size, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.W1[i] = torch.randn(
                    self.n_dims, hidden_layer_size, generator=generator
                )
                self.W2[i] = torch.randn(hidden_layer_size, 1, generator=generator)
        else:
            assert "W1" in pool_dict and "W2" in pool_dict
            assert len(pool_dict["W1"]) == len(pool_dict["W2"])
            indices = torch.randperm(len(pool_dict["W1"]))[:batch_size]
            self.W1 = pool_dict["W1"][indices]
            self.W2 = pool_dict["W2"][indices]

    def evaluate(self, xs_b):
        W1 = self.W1.to(xs_b.device)
        W2 = self.W2.to(xs_b.device)
        # Renormalize to Linear Regression Scale
        ys_b_nn = (torch.nn.functional.relu(xs_b @ W1) @ W2)[:, :, 0]
        ys_b_nn = ys_b_nn * math.sqrt(2 / self.hidden_layer_size)
        ys_b_nn = self.scale * ys_b_nn
        #         ys_b_nn = ys_b_nn * math.sqrt(self.n_dims) / ys_b_nn.std()
        return ys_b_nn

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        return {
            "W1": torch.randn(num_tasks, n_dims, hidden_layer_size),
            "W2": torch.randn(num_tasks, hidden_layer_size, 1),
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class DecisionTree(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, depth=4):

        super(DecisionTree, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.depth = depth

        if pool_dict is None:

            # We represent the tree using an array (tensor). Root node is at index 0, its 2 children at index 1 and 2...
            # dt_tensor stores the coordinate used at each node of the decision tree.
            # Only indices corresponding to non-leaf nodes are relevant
            self.dt_tensor = torch.randint(
                low=0, high=n_dims, size=(batch_size, 2 ** (depth + 1) - 1)
            )

            # Target value at the leaf nodes.
            # Only indices corresponding to leaf nodes are relevant.
            self.target_tensor = torch.randn(self.dt_tensor.shape)
        elif seeds is not None:
            self.dt_tensor = torch.zeros(batch_size, 2 ** (depth + 1) - 1)
            self.target_tensor = torch.zeros_like(dt_tensor)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.dt_tensor[i] = torch.randint(
                    low=0,
                    high=n_dims - 1,
                    size=2 ** (depth + 1) - 1,
                    generator=generator,
                )
                self.target_tensor[i] = torch.randn(
                    self.dt_tensor[i].shape, generator=generator
                )
        else:
            raise NotImplementedError

    def evaluate(self, xs_b):
        dt_tensor = self.dt_tensor.to(xs_b.device)
        target_tensor = self.target_tensor.to(xs_b.device)
        ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i in range(xs_b.shape[0]):
            xs_bool = xs_b[i] > 0
            # If a single decision tree present, use it for all the xs in the batch.
            if self.b_size == 1:
                dt = dt_tensor[0]
                target = target_tensor[0]
            else:
                dt = dt_tensor[i]
                target = target_tensor[i]

            cur_nodes = torch.zeros(xs_b.shape[1], device=xs_b.device).long()
            for j in range(self.depth):
                cur_coords = dt[cur_nodes]
                cur_decisions = xs_bool[torch.arange(xs_bool.shape[0]), cur_coords]
                cur_nodes = 2 * cur_nodes + 1 + cur_decisions

            ys_b[i] = target[cur_nodes]

        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class GaussianKernelRegression(Task):
    """
    Represents a function of the form:
        f(x) = sum_{i=1}^{R} alpha_i * exp( -|| x - c_i ||^2 / (2*sigma^2) ).
    The 'centers' c_i and 'alphas' alpha_i are sampled either from random
    normal distributions, or from a pre-defined pool_dict, or using seeds
    for deterministic generation.

    Args:
        n_dims: input dimension
        batch_size: number of tasks in each batch
        pool_dict: optional dictionary that stores pre-generated centers/alphas
        seeds: optional list of seeds to generate deterministic data
        R: number of Gaussian kernel centers
        sigma: kernel bandwidth
        scale: an optional scale factor on top of the final sum
        valid_coords: if we want to only use part of the dimension for the centers
                      (similar to "curriculum" or "sparse" style).
    """

    def __init__(self,
                 n_dims,
                 batch_size,
                 pool_dict=None,
                 seeds=None,
                 R=10,
                 sigma=1.0,
                 scale=1.0,
                 valid_coords=None):
        super(GaussianKernelRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)

        self.R = R
        self.sigma = sigma
        self.scale = scale
        self.valid_coords = valid_coords if valid_coords is not None else n_dims

        self.centers = torch.zeros(self.b_size, R, n_dims)
        self.alphas = torch.zeros(self.b_size, R)

        if pool_dict is None and seeds is None:
            self._random_init()
        elif seeds is not None:
            if len(seeds) != self.b_size:
                raise ValueError(f"Expected {self.b_size} seeds, got {len(seeds)}")
            self._init_with_seeds(seeds)
        elif pool_dict is not None:
            self._init_from_pool(pool_dict)
        else:
            raise ValueError("Unhandled combination of pool_dict and seeds")

    def _random_init(self):
        self.centers = torch.randn(self.b_size, self.R, self.n_dims)
        self.alphas = torch.randn(self.b_size, self.R)
        self._truncate_dimensions()

    def _init_with_seeds(self, seeds):
        for i, seed in enumerate(seeds):
            generator = torch.Generator()
            generator.manual_seed(seed)

            tmp_centers = torch.randn(self.R, self.n_dims, generator=generator)
            tmp_alphas = torch.randn(self.R, generator=generator)

            self.centers[i] = tmp_centers
            self.alphas[i] = tmp_alphas
        self._truncate_dimensions()

    def _init_from_pool(self, pool_dict):
        C_all = pool_dict["centers"]
        A_all = pool_dict["alphas"]

        if len(C_all) != len(A_all):
            raise ValueError("pool_dict mismatch: centers and alphas differ in length")

        if len(C_all) < self.b_size:
            raise ValueError("pool_dict not large enough for this batch_size")

        indices = torch.randperm(len(C_all))[: self.b_size]
        self.centers = C_all[indices].clone()
        self.alphas  = A_all[indices].clone()
        self._truncate_dimensions()

    def _truncate_dimensions(self):
        if self.valid_coords < self.n_dims:
            self.centers[:, :, self.valid_coords:] = 0

    def evaluate(self, xs_b):
        device = xs_b.device
        b_size, n_points, _ = xs_b.shape

        centers_b = self.centers.to(device)
        alphas_b = self.alphas.to(device)
        valid_coords = self.valid_coords

        ys_b = torch.zeros(b_size, n_points, device=device)

        for b in range(b_size):
            c_b = centers_b[b][:, :valid_coords]
            xs_valid = xs_b[b][:, :valid_coords]

            diff = xs_valid.unsqueeze(1) - c_b.unsqueeze(0)
            dist_sq = (diff ** 2).sum(dim=2)

            kernel_vals = torch.exp(-dist_sq / (2.0 * self.sigma**2))
            y_b = (kernel_vals * alphas_b[b]).sum(dim=1)
            ys_b[b] = self.scale * y_b

        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, R=10, sigma=1.0, scale=1.0, **kwargs):
        centers = torch.randn(num_tasks, R, n_dims)
        alphas = torch.randn(num_tasks, R)

        pool_dict = {
            "centers": centers,
            "alphas": alphas,
        }
        return pool_dict

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class Example1Task(Task):
    """
    Implements Example 1 from the paper: 10-dimensional input, first 4 dimensions are effective signals.
    f(x) = 5*g1(x^(1)) + 3*g2(x^(2)) + 4*g3(x^(3)) + 6*g4(x^(4)) + noise
    where:
      g1(t) = t
      g2(t) = (2t - 1)^2
      g3(t) = sin(2πt) / (2 - sin(2πt))
      g4(t) = 0.1 sin(2πt) + 0.2 cos(2πt) + 0.3 sin^2(2πt)
              + 0.4 cos^3(2πt) + 0.5 sin^3(2πt)
    """

    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None,
                 noise_std=1.74,  # Original paper uses ~1.74 for SNR=3:1
                 **kwargs):
        """
        Args:
            noise_std: Standard deviation of noise, ~1.74 in original paper for SNR=3:1
        """
        super().__init__(n_dims, batch_size, pool_dict, seeds)
        self.noise_std = noise_std

        # Define the four basis functions
        def g1(t):
            return t
        
        def g2(t):
            return (2.0 * t - 1.0) ** 2
        
        def g3(t):
            return torch.sin(2.0 * math.pi * t) / (2.0 - torch.sin(2.0 * math.pi * t))
        
        def g4(t):
            return (0.1 * torch.sin(2.0 * math.pi * t)
                    + 0.2 * torch.cos(2.0 * math.pi * t)
                    + 0.3 * torch.sin(2.0 * math.pi * t)**2
                    + 0.4 * torch.cos(2.0 * math.pi * t)**3
                    + 0.5 * torch.sin(2.0 * math.pi * t)**3)

        self.g_funcs = [g1, g2, g3, g4]
        self.coefs = [5.0, 3.0, 4.0, 6.0]

    def evaluate(self, xs_b: torch.Tensor) -> torch.Tensor:
        """
        xs_b: shape = [batch_size, n_points, n_dims], where n_dims=10
        Returns ys_b: shape = [batch_size, n_points]
        """
        device = xs_b.device
        b_size, n_points, d = xs_b.shape

        # Ensure d>=4 as we need first 4 dims to compute f(x)
        assert d >= 4, f"Expected at least 4 dims, got d={d}"

        ys_b = torch.zeros(b_size, n_points, device=device)

        for i in range(b_size):
            # Get all x for current sample (n_points, d)
            x_i = xs_b[i]
            # Get first 4 dimensions
            x1, x2, x3, x4 = x_i[:, 0], x_i[:, 1], x_i[:, 2], x_i[:, 3]

            # Compute value term by term
            val = torch.zeros(n_points, device=device)
            for func, c, xcol in zip(self.g_funcs, self.coefs, [x1, x2, x3, x4]):
                val += c * func(xcol)

            # Add noise
            noise = torch.randn(n_points, device=device) * self.noise_std
            val += noise

            ys_b[i] = val

        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        # This example doesn't need any pre-generated parameters
        return None

    @staticmethod
    def get_metric():
        # Use squared error
        return squared_error

    @staticmethod
    def get_training_metric():
        # Use mean squared error for training
        return mean_squared_error

class Example2Task(Task):
    """
    Implements Example 2 from the paper:
    60-dimensional input，总共有 12 个有效变量（前 12 维），并且每 4 维对应和 Example1 相同的 g1,g2,g3,g4 函数。
    其函数形式为:
      f(x) = g1(x1) + g2(x2) + g3(x3) + g4(x4)
           + 1.5*g1(x5) + 1.5*g2(x6) + 1.5*g3(x7) + 1.5*g4(x8)
           + 2*g1(x9) + 2*g2(x10) + 2*g3(x11) + 2*g4(x12)
    并在其基础上加高斯噪声 (noise_std ~ 0.72).
    剩余 x13,...,x60 均为无用变量(不影响 f ).
    """

    def __init__(self, 
                 n_dims, 
                 batch_size,
                 pool_dict=None, 
                 seeds=None,
                 # 这里的 0.72≈sqrt(0.5184),对标原文SNR=3:1
                 noise_std=0.72,  
                 **kwargs):
        super().__init__(n_dims, batch_size, pool_dict, seeds)
        self.noise_std = noise_std

        # 跟 Example1 里相同的四个基函数:
        def g1(t):
            return t

        def g2(t):
            return (2.0 * t - 1.0) ** 2

        def g3(t):
            return torch.sin(2.0 * math.pi * t) / (2.0 - torch.sin(2.0 * math.pi * t))

        def g4(t):
            return (0.1 * torch.sin(2.0 * math.pi * t)
                    + 0.2 * torch.cos(2.0 * math.pi * t)
                    + 0.3 * (torch.sin(2.0 * math.pi * t))**2
                    + 0.4 * (torch.cos(2.0 * math.pi * t))**3
                    + 0.5 * (torch.sin(2.0 * math.pi * t))**3)

        # 为了方便，我们按数组存储
        self.g_funcs = [g1, g2, g3, g4]  # 基础函数
        # 每组系数: [1, 1, 1, 1], [1.5, 1.5, 1.5, 1.5], [2, 2, 2, 2]
        self.amplitudes = [
            [1.0,  1.0,  1.0,  1.0 ],
            [1.5,  1.5,  1.5,  1.5 ],
            [2.0,  2.0,  2.0,  2.0 ],
        ]

    def evaluate(self, xs_b: torch.Tensor) -> torch.Tensor:
        """
        xs_b: shape = [batch_size, n_points, n_dims], 这里 n_dims=60
        Returns ys_b: shape = [batch_size, n_points]
        """
        device = xs_b.device
        b_size, n_points, d = xs_b.shape

        # 原文 Example 2 假设 d=60，但只用到前12维
        assert d >= 12, f"Example2Task expects at least 12 dims, got d={d}"

        ys_b = torch.zeros(b_size, n_points, device=device)

        for i in range(b_size):
            x_i = xs_b[i]  # shape=(n_points, d)
            # 先取 x1~x12
            # block1: x[0:4], block2: x[4:8], block3: x[8:12]
            # 与 4 个函数 g1,g2,g3,g4 配对
            val = torch.zeros(n_points, device=device)

            # 第 1 组: amplitudes = [1,1,1,1]
            for j in range(4):
                func = self.g_funcs[j]
                amp  = self.amplitudes[0][j]
                xcol = x_i[:, j]   # x(1), x(2), x(3), x(4)
                val += amp * func(xcol)

            # 第 2 组: amplitudes = [1.5,1.5,1.5,1.5]
            for j in range(4):
                func = self.g_funcs[j]
                amp  = self.amplitudes[1][j]
                xcol = x_i[:, 4 + j]  # x(5)~x(8)
                val += amp * func(xcol)

            # 第 3 组: amplitudes = [2.0,2.0,2.0,2.0]
            for j in range(4):
                func = self.g_funcs[j]
                amp  = self.amplitudes[2][j]
                xcol = x_i[:, 8 + j]  # x(9)~x(12)
                val += amp * func(xcol)

            # 后面 x(13)~x(60) 不影响 f(x)

            # 加上高斯噪声
            noise = torch.randn(n_points, device=device) * self.noise_std
            val += noise

            ys_b[i] = val

        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        # 跟 Example1 类似，如果不需要 pool_dict 可直接返回 None
        return None

    @staticmethod
    def get_metric():
        return squared_error  # 用 squared error 评估

    @staticmethod
    def get_training_metric():
        return mean_squared_error