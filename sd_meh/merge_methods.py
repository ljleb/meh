import math
from typing import Tuple

import torch
from torch import Tensor

__all__ = [
    "weighted_sum",
    "weighted_subtraction",
    "tensor_sum",
    "add_difference",
    "sum_twice",
    "triple_sum",
    "euclidean_add_difference",
    "multiply_difference",
    "top_k_tensor_sum",
    "similarity_add_difference",
    "distribution_crossover",
    "ties_add_difference",
]


EPSILON = 1e-10  # Define a small constant EPSILON to prevent division by zero


def weighted_sum(a: Tensor, b: Tensor, alpha: float, **kwargs) -> Tensor:
    return (1 - alpha) * a + alpha * b


def weighted_subtraction(
    a: Tensor, b: Tensor, alpha: float, beta: float, **kwargs
) -> Tensor:
    # Adjust beta if both alpha and beta are 1.0 to avoid division by zero
    if alpha == 1.0 and beta == 1.0:
        beta -= EPSILON

    return (a - alpha * beta * b) / (1 - alpha * beta)


def tensor_sum(a: Tensor, b: Tensor, alpha: float, beta: float, **kwargs) -> Tensor:
    if alpha + beta <= 1:
        tt = a.clone()
        talphas = int(a.shape[0] * beta)
        talphae = int(a.shape[0] * (alpha + beta))
        tt[talphas:talphae] = b[talphas:talphae].clone()
    else:
        talphas = int(a.shape[0] * (alpha + beta - 1))
        talphae = int(a.shape[0] * beta)
        tt = b.clone()
        tt[talphas:talphae] = a[talphas:talphae].clone()
    return tt


def add_difference(a: Tensor, b: Tensor, c: Tensor, alpha: float, **kwargs) -> Tensor:
    return a + alpha * (b - c)


def sum_twice(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, **kwargs
) -> Tensor:
    return (1 - beta) * ((1 - alpha) * a + alpha * b) + beta * c


def triple_sum(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, **kwargs
) -> Tensor:
    return (1 - alpha - beta) * a + alpha * b + beta * c


def euclidean_add_difference(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, **kwargs
) -> Tensor:
    a_diff = a.float() - c.float()
    b_diff = b.float() - c.float()
    a_diff = torch.nan_to_num(a_diff / torch.linalg.norm(a_diff))
    b_diff = torch.nan_to_num(b_diff / torch.linalg.norm(b_diff))

    distance = (1 - alpha) * a_diff**2 + alpha * b_diff**2
    distance = torch.sqrt(distance)
    sum_diff = weighted_sum(a.float(), b.float(), alpha) - c.float()
    distance = torch.copysign(distance, sum_diff)

    target_norm = torch.linalg.norm(sum_diff)
    return c + distance / torch.linalg.norm(distance) * target_norm


def multiply_difference(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, **kwargs
) -> Tensor:
    diff_a = torch.pow(torch.abs(a.float() - c), (1 - alpha))
    diff_b = torch.pow(torch.abs(b.float() - c), alpha)
    difference = torch.copysign(diff_a * diff_b, weighted_sum(a, b, beta) - c)
    return c + difference.to(c.dtype)


def top_k_tensor_sum(
    a: Tensor, b: Tensor, alpha: float, beta: float, **kwargs
) -> Tensor:
    a_flat = torch.flatten(a)
    a_dist = torch.msort(a_flat)
    b_indices = torch.argsort(torch.flatten(b), stable=True)
    redist_indices = torch.argsort(b_indices)

    start_i, end_i, region_is_inverted = ratio_to_region(alpha, beta, torch.numel(a))
    start_top_k = kth_abs_value(a_dist, start_i)
    end_top_k = kth_abs_value(a_dist, end_i)

    indices_mask = (start_top_k < torch.abs(a_dist)) & (torch.abs(a_dist) <= end_top_k)
    if region_is_inverted:
        indices_mask = ~indices_mask
    indices_mask = torch.gather(indices_mask.float(), 0, redist_indices)

    a_redist = torch.gather(a_dist, 0, redist_indices)
    a_redist = (1 - indices_mask) * a_flat + indices_mask * a_redist
    return a_redist.reshape_as(a)


def kth_abs_value(a: Tensor, k: int) -> Tensor:
    if k <= 0:
        return torch.tensor(-1, device=a.device)
    else:
        return torch.kthvalue(torch.abs(a.float()), k)[0]


def ratio_to_region(width: float, offset: float, n: int) -> Tuple[int, int, bool]:
    if width < 0:
        offset += width
        width = -width
    width = min(width, 1)

    if offset < 0:
        offset = 1 + offset - int(offset)
    offset = math.fmod(offset, 1.0)

    if width + offset <= 1:
        inverted = False
        start = offset * n
        end = (width + offset) * n
    else:
        inverted = True
        start = (width + offset - 1) * n
        end = offset * n

    return round(start), round(end), inverted


def similarity_add_difference(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, **kwargs
) -> Tensor:
    threshold = torch.maximum(torch.abs(a), torch.abs(b))
    similarity = ((a * b / threshold**2) + 1) / 2
    similarity = torch.nan_to_num(similarity * beta, nan=beta)

    ab_diff = a + alpha * (b - c)
    ab_sum = (1 - alpha / 2) * a + (alpha / 2) * b
    return (1 - similarity) * ab_diff + similarity * ab_sum


def distribution_crossover(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, **kwargs
):
    if a.shape == ():
        return alpha * a + (1 - alpha) * b

    c_indices = torch.argsort(torch.flatten(c))
    a_dist = torch.gather(torch.flatten(a), 0, c_indices)
    b_dist = torch.gather(torch.flatten(b), 0, c_indices)

    a_dft = torch.fft.rfft(a_dist.float())
    b_dft = torch.fft.rfft(b_dist.float())

    dft_filter = create_filter((torch.numel(a_dft),), alpha, beta, device=a.device)

    x_dft = (1 - dft_filter) * a_dft + dft_filter * b_dft
    x_dist = torch.fft.irfft(x_dft, a_dist.shape[0])
    x_values = torch.gather(x_dist, 0, torch.argsort(c_indices))
    return x_values.reshape_as(a)


def crossover(
    a: Tensor, b: Tensor, alpha: float, beta: float, **kwargs
):
    # if alpha == 0 and beta == 0:
    #     return a

    if len(a.shape) == 0 or torch.allclose(a.half(), b.half()):
        return weighted_sum(a, b, beta)

    a_dft = torch.fft.rfftn(a.float())
    b_dft = torch.fft.rfftn(b.float())

    dft_filter = create_filter(a.shape, alpha, beta, device=a.device)

    x_dft = (1 - dft_filter) * a_dft + dft_filter * b_dft
    x = torch.fft.irfftn(x_dft, s=a.shape)
    return x.to(a.dtype)


def create_filter(shape, alpha: float, beta: float, device=None):
    # Generate linear indices for each dimension
    gradients = [
        torch.linspace(0, 1, s, device=device)**2
        for s in shape[:-1]
    ] + [torch.linspace(0, 1, shape[-1] // 2 + 1, device=device)**2]

    if len(shape) > 1:
        grids = torch.meshgrid(*gradients, indexing='ij')
        linear_scale = torch.sqrt(torch.sum(torch.stack(grids), dim=0))
    else:
        linear_scale = gradients[0]

    if beta < EPSILON:
        return (linear_scale <= alpha).float()

    b = math.pi * min(beta, 1-beta) / 2
    d1 = math.sin(b)
    d2 = math.cos(b)
    d3 = d1 + d2

    p1_d1 = (1 - math.sqrt(2) * math.cos(2*b + math.pi/4)) / 4
    p2_d2 = (math.sin(2*b) + 3*math.cos(2*b) + 1) / 4
    p3_d3 = (math.sqrt(2) * math.sin(2*b + math.pi/4) + 1) / 2

    def cot(x):
        return 1/math.tan(x)

    a = min(max(1 - alpha, 0.0), 1.0) * p3_d3
    if a < p1_d1:
        a = math.sqrt(2 * a / (cot(b) + 1))
    elif a < p2_d2:
        a = (math.sqrt(2)*(4*a + 1) - 2*math.cos(2*b + math.pi/4)) / (8*math.sin(b + math.pi / 4))
    else:
        b_sin2 = math.sin(2*b)
        a = (b_sin2 + 1)/d3 - math.sqrt((a*(math.cos(2*b) - 1) + (b_sin2 - a + 1)*b_sin2) / (b_sin2 + 1))
    a /= d3

    # Apply alpha and beta
    b_tan = cot(b) if beta <= 0.5 else math.tan(b)
    adjusted_filter = linear_scale * b_tan - a * b_tan - a + 1
    adjusted_filter = torch.nan_to_num(torch.clamp(adjusted_filter, 0.0, 1.0))

    return adjusted_filter


def ties_add_difference(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, **kwargs
) -> Tensor:
    deltas = []
    signs = []
    for m in [a, b]:
        deltas.append(filter_top_k(m - c, beta))
        signs.append(torch.sign(deltas[-1]))

    signs = torch.stack(signs, dim=0)
    final_sign = torch.sign(torch.sum(signs, dim=0))
    delta_filters = (signs == final_sign).float()

    res = torch.zeros_like(c, device=c.device)
    for delta_filter, delta in zip(delta_filters, deltas):
        res += delta_filter * delta

    param_count = torch.sum(delta_filters, dim=0)
    return c + alpha * torch.nan_to_num(res / param_count)


def filter_top_k(a: Tensor, k: float):
    k = max(int((1 - k) * torch.numel(a)), 1)
    k_value, _ = torch.kthvalue(torch.abs(a.flatten()).float(), k)
    top_k_filter = (torch.abs(a) >= k_value).float()
    return a * top_k_filter
