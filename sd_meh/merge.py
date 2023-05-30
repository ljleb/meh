import math
import os
import re
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy
import safetensors.torch
import scipy
import torch
from tqdm import tqdm

from sd_meh.model import SDModel

MAX_TOKENS = 77
NUM_INPUT_BLOCKS = 12
NUM_MID_BLOCK = 1
NUM_OUTPUT_BLOCKS = 12
NUM_TOTAL_BLOCKS = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + NUM_OUTPUT_BLOCKS
EPSILON = 1e-10  # Define a small constant EPSILON to prevent division by zero

KEY_POSITION_IDS = ".".join(
    [
        "cond_stage_model",
        "transformer",
        "text_model",
        "embeddings",
        "position_ids",
    ]
)


NAI_KEYS = {
    "cond_stage_model.transformer.embeddings.": "cond_stage_model.transformer.text_model.embeddings.",
    "cond_stage_model.transformer.encoder.": "cond_stage_model.transformer.text_model.encoder.",
    "cond_stage_model.transformer.final_layer_norm.": "cond_stage_model.transformer.text_model.final_layer_norm.",
}


def fix_clip(model: Dict) -> Dict:
    if KEY_POSITION_IDS in model:
        model[KEY_POSITION_IDS] = torch.tensor(
            [list(range(MAX_TOKENS))], dtype=torch.int64
        )

    return model


def fix_key(model: Dict, key: str) -> Dict:
    for nk in NAI_KEYS:
        if key.startswith(nk):
            model[key.replace(nk, NAI_KEYS[nk])] = model[key]
            del model[key]

    return model


# https://github.com/j4ded/sdweb-merge-block-weighted-gui/blob/master/scripts/mbw/merge_block_weighted.py#L115
def fix_model(model: Dict) -> Dict:
    for k in model:
        model = fix_key(model, k)
    return fix_clip(model)


def load_sd_model(model: os.PathLike | str, device: str = "cpu") -> Dict:
    if isinstance(model, str):
        model = Path(model)

    return SDModel(model, device).load_model()


def merge_models(
    models: Dict[str, os.PathLike | str],
    weights: Dict,
    bases: Dict,
    merge_mode: str,
    precision: int = 16,
    weights_clip: Optional[float] = None,
) -> Dict:
    thetas = {k: load_sd_model(m) for k, m in models.items()}

    for key in tqdm(thetas["model_a"].keys(), desc="stage 1"):
        if result := merge_key(
            key,
            thetas,
            weights,
            bases,
            merge_mode,
            precision,
            weights_clip,
        ):
            thetas["model_a"][key] = result[1]

    for key in tqdm(thetas["model_b"].keys(), desc="stage 2"):
        if KEY_POSITION_IDS in key:
            continue
        if "model" in key and key not in thetas["model_a"]:
            thetas["model_a"].update({key: thetas["model_b"][key]})
            if precision == 16:
                thetas["model_a"][key] = thetas["model_a"][key].half()

    return fix_model(thetas["model_a"])


def merge_key(
    key: str,
    thetas: Dict,
    weights: Dict,
    bases: Dict,
    merge_mode: str,
    precision: int = 16,
    weights_clip: Optional[float] = None,
) -> Optional[Tuple[str, Dict]]:
    if KEY_POSITION_IDS in key:
        return

    for theta in thetas.values():
        if key not in theta:
            return

    if "model" in key:
        current_bases = bases

        if "model.diffusion_model." in key:
            weight_index = -1

            re_inp = re.compile(r"\.input_blocks\.(\d+)\.")  # 12
            re_mid = re.compile(r"\.middle_block\.(\d+)\.")  # 1
            re_out = re.compile(r"\.output_blocks\.(\d+)\.")  # 12

            if "time_embed" in key:
                weight_index = 0  # before input blocks
            elif ".out." in key:
                weight_index = NUM_TOTAL_BLOCKS - 1  # after output blocks
            elif m := re_inp.search(key):
                weight_index = int(m.groups()[0])
            elif re_mid.search(key):
                weight_index = NUM_INPUT_BLOCKS
            elif m := re_out.search(key):
                weight_index = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + int(m.groups()[0])

            if weight_index >= NUM_TOTAL_BLOCKS:
                raise ValueError(f"illegal block index {key}")

            if weight_index >= 0:
                current_bases = {k: w[weight_index] for k, w in weights.items()}

        merged_key = merge(current_bases, thetas, key, merge_mode)

        if weights_clip is not None:
            t0 = thetas["model_a"][key]
            t1 = thetas["model_b"][key]
            t2 = thetas["model_c"][key]
            t0 = torch.where(torch.isnan(t0), torch.where(torch.isnan(t1), t2, t1), t0)
            t1 = torch.where(torch.isnan(t1), torch.where(torch.isnan(t2), t0, t2), t1)
            t2 = torch.where(torch.isnan(t2), torch.where(torch.isnan(t0), t1, t0), t2)
            threshold = torch.maximum(torch.abs(t0 - t2), torch.abs(t1 - t2)) #* 1.0015
            merged_key = t2 + clip_key(merged_key - t2, threshold, weights_clip)

        if precision == 16:
            merged_key = merged_key.half()

        return key, merged_key


def clip_key(theta, threshold, soft_amount):
    soft_amount /= 10
    if soft_amount <= EPSILON:
        return torch.minimum(torch.maximum(theta, -threshold), threshold)

    def softplus(offset_t):
        res = torch.log(1 + torch.exp(torch.nan_to_num(offset_t.float() / soft_amount / threshold))) * soft_amount * threshold
        res = torch.where(offset_t.float() / soft_amount / threshold > 20, offset_t, res)
        res = torch.where(soft_amount * threshold == 0, threshold, res)
        assert not torch.isnan(res).any()
        return res

    return softplus(threshold + theta) - softplus(threshold - theta) - theta


def merge(current_bases: Dict, thetas: Dict, key: str, merge_mode: str) -> torch.Tensor:
    t0 = thetas["model_a"][key]
    t1 = thetas["model_b"][key]
    alpha = current_bases["alpha"]
    if merge_mode == "weighted_sum":
        return (1 - alpha) * t0 + alpha * t1
    elif merge_mode == "weighted_subtraction":
        beta = current_bases["beta"]
        # Adjust beta if both alpha and beta are 1.0 to avoid division by zero
        if alpha == 1.0 and beta == 1.0:
            beta -= EPSILON
        return (t0 - alpha * beta * t1) / (1 - alpha * beta)
    elif merge_mode == "tensor_sum":
        beta = current_bases["beta"]
        if alpha + beta <= 1:
            tt = t0.clone()
            talphas = int(t0.shape[0] * (beta))
            talphae = int(t0.shape[0] * (alpha + beta))
            tt[talphas:talphae] = t1[talphas:talphae].clone()
        else:
            talphas = int(t0.shape[0] * (alpha + beta - 1))
            talphae = int(t0.shape[0] * (beta))
            tt = t1.clone()
            tt[talphas:talphae] = t0[talphas:talphae].clone()
        return tt
    elif merge_mode == "blur":
        if t0.shape == ():
            return t0
        mgrid = numpy.mgrid[(slice(-1.0, 1.0, 3j),) * len(t0.shape)]
        coords = numpy.column_stack([x.flat for x in mgrid])
        mu = numpy.array([0.0] * len(t0.shape))
        sigma = numpy.array([.025] * len(t0.shape))
        covariance = numpy.diag(sigma ** 2)
        z = scipy.stats.multivariate_normal.pdf(coords, mean=mu, cov=covariance)
        z = z.reshape(mgrid[0].shape)
        res = t1 + torch.Tensor(scipy.signal.fftconvolve(t0 - t1, z, mode='same'))
        return res

    t2 = thetas["model_c"][key]
    if merge_mode == "add_difference":
        return t0 + alpha * (t1 - t2)
    elif merge_mode == "convolve":
        if t0.shape == ():
            return torch.nan_to_num(t0 * t1 / t2)
        dft = torch.fft.rfftn(t0.float()) * torch.fft.rfftn(t1.float()) / torch.fft.rfftn(t2.float())
        res = torch.fft.irfftn(dft, t0.size())
        return res
    elif merge_mode == "add_highpass":
        if t0.shape == ():
            return t0
        dft = torch.fft.rfftn(t1.float() - t2.float())
        filter_rate = current_bases["beta"]
        rshape = tuple(slice(int(size * filter_rate), size) for size in dft.shape)
        dft[rshape] = 0
        res = t0 + alpha * torch.fft.irfftn(dft, t1.size())
        threshold = torch.maximum(torch.abs(t1), torch.abs(t2))
        res = torch.minimum(torch.maximum(res, -threshold), threshold)
        return res
    elif merge_mode == "similarity_add_difference":
        threshold = torch.maximum(torch.abs(t0), torch.abs(t1))
        similarity = (1 - (t0 * t1 / threshold ** 2) ** 2) / 2 * alpha
        similarity = torch.nan_to_num(similarity, nan=alpha)
        res = t0 + similarity * (2 * t1 - t2)
        res = torch.minimum(torch.maximum(res, -threshold), threshold)
        return res
    elif merge_mode == "hardclip":
        threshold = torch.maximum(torch.abs(t1), torch.abs(t2))
        return torch.minimum(torch.maximum(t0, -threshold), threshold)
    elif merge_mode == "euclidian_add_difference":
        dist = (t0 - t2) ** 2 + alpha * (t1 - t2) ** 2
        try:
            dist = torch.sqrt(dist)
        except RuntimeError:
            dist = torch.sqrt(dist.float()).half()
        dist = torch.copysign(dist, t0 + t1 - 2 * t2)
        norm = (torch.linalg.norm(t0 - t2) + torch.linalg.norm(t1 - t2)) / 2
        return dist / torch.linalg.norm(dist) * norm + t2
    elif merge_mode == "euclidian_weighted_sum": # old
        dist = (1 - alpha) * (t0 - t2) ** 2 + alpha * (t1 - t2) ** 2
        try:
            dist = torch.sqrt(dist)
        except RuntimeError:
            dist = torch.sqrt(dist.float()).half()
        res = torch.copysign(dist, (1 - alpha) * t0 + alpha * t1 - t2) + t2
        norm = (torch.linalg.norm(t0) + torch.linalg.norm(t1)) / 2
        return res / torch.linalg.norm(res) * norm
    elif merge_mode == "sigmoid_add_difference":
        dif_res = alpha * (t1 - t2)
        threshold = torch.maximum(torch.abs(t0 - t2), torch.abs(t1 - t2))
        return t0 + (torch.sigmoid(dif_res / threshold * 4) * 2 - 1) * threshold
    elif merge_mode == "tanh_add_difference":
        res = alpha * (t1 - t2)
        threshold = torch.maximum(torch.abs(t0 - t2), torch.abs(t1 - t2))
        res = torch.minimum(torch.maximum(res, -threshold * 1.5), threshold * 1.5)
        return t0 + torch.tanh(torch.tan(res / threshold)) * threshold
    elif merge_mode == "multiply_difference":
        t3 = thetas["model_d"][key]
        return (t2 + t3) / 2 + torch.copysign(torch.sqrt(torch.abs((t0 - t2) * (t1 - t3)).float()), t0 + t1 - t2 - t3)
    elif merge_mode == "relative_softplus":
        #soft_plus = torch.nn.Softplus(beta=8)
        #soft_plus = torch.nn.Softplus(beta=32, threshold=8)
        soft_plus = torch.nn.Softplus(beta=16, threshold=16)
        threshold = torch.maximum(torch.abs(t0), torch.abs(t1))
        res = alpha * (t1 - t2)
        hard_clip = torch.minimum(torch.maximum(res, -threshold), threshold)
        res /= threshold
        soft_clip = (soft_plus(1 + res) - soft_plus(1 - res) - res) * threshold
        del res
        similarity = (t0 * t1 / threshold + 1) / 2
        return t0 + (1 - similarity) * hard_clip + similarity * soft_clip
    elif merge_mode == "max_sum":
        max_threshold = torch.copysign(torch.maximum(torch.abs(t0 - t2), torch.abs(t1 - t2)), t0 + t1 - 2 * t2)
        return t2 + max_threshold
    elif merge_mode == "sq_mean":
        tx = t0 + t1 - t2
        mean = torch.mean(tx)
        return (tx - mean) ** 2 + mean
    elif merge_mode == "edit_dist":
        t0_values, t0_indices = torch.sort(torch.flatten(t0.cuda()))
        t1_indices = torch.argsort(torch.flatten(t1.cuda()), stable=True)
        redistributed_t0_values = torch.gather(t0_values, 0, t1_indices)
        return redistributed_t0_values.reshape(t0.shape)
    threshold = torch.maximum(torch.abs(t0), torch.abs(t1))
    beta = current_bases["beta"]
    if merge_mode == "similarity_add_difference":
        res_0 = t0 + t1 - t2
        res_1 = (t0 + t1) / 2

        similarity = ((t0 * t1 / threshold ** 2) + 1) / 2
        similarity = torch.nan_to_num(similarity * beta, nan=beta)
        return (1 - similarity) * res_0 + similarity * res_1
    elif merge_mode == "redistribute":
        t0_values = torch.msort(torch.flatten(t0))
        t1_indices = torch.argsort(torch.flatten(t1), stable=True)
        redistributed_t0_values = torch.gather(t0_values, 0, torch.argsort(t1_indices))
        return redistributed_t0_values.reshape(t0.shape)
    elif merge_mode == "softdist":
        if t0.shape == ():
            return t0

        t0_indices = torch.argsort(torch.flatten(t0.cuda()))
        t0_dist = torch.gather(torch.flatten(t0.cuda()), 0, t0_indices)
        t0_dft = torch.fft.rfftn(t0_dist.float())
        t0_dft[int(alpha * t0_dft.size(0)):] = 0
        t0_dist = torch.fft.irfftn(t0_dft, t0_dist.size())
        redistributed_t0_values = torch.gather(t0_dist, 0, torch.argsort(t0_indices))
        return redistributed_t0_values.reshape(t0.shape)
    elif merge_mode == "from_linear":
        tx_values = (torch.arange(0, torch.numel(t0)) / torch.numel(t0)) * (torch.max(t0) - torch.min(t0)) + torch.min(t0)
        t0_indices = torch.argsort(torch.flatten(t0), stable=True)
        redistributed_tx_values = torch.gather(tx_values, 0, torch.argsort(t0_indices))
        return redistributed_tx_values.reshape(t0.shape)
    t2 = thetas["model_c"][key]
    if merge_mode == "add_difference":
        return t0 + alpha * (t1 - t2)
    elif merge_mode == "clip":
        if t0.shape == ():
            return t0

        dft = torch.fft.rfftn((t0 - 2 * t1 + t2).float())
        return t0
    if merge_mode == "similarity_add_difference":
        res_0 = t0 + t1 - t2
        res_1 = (t0 + t1) / 2

        threshold = torch.maximum(torch.abs(t0), torch.abs(t1))
        similarity = (((t0 * t1 / threshold ** 2) + 1) / 2) ** 2 * alpha
        similarity = torch.nan_to_num(similarity, nan=alpha)
        return (1 - similarity) * res_0 + similarity * res_1
    elif merge_mode == "add_similarity":
        sim = (t0 * t1 / torch.maximum(torch.abs(t0), torch.abs(t1)) ** 2 / 2 + 1) / 2
        #sim12 = (t1 * t2 / torch.maximum(torch.abs(t1), torch.abs(t2)) ** 2 + 1) / 2
        return t0 + alpha * sim.float() * (t1 - t2)
    elif merge_mode == "distribution_softclip":
        if t0.shape == ():
            return t0

        # t2_dist = torch.gather(torch.flatten(t2.cuda()), 0, torch.argsort(torch.flatten(t2.cuda())))
        # tx_values = torch.gather(t2_dist, 0, torch.argsort(torch.argsort(torch.flatten(t0 + t1 - t2).cuda())))
        # return tx_values.reshape(t0.shape).cpu()

        # t0_indices = torch.argsort(torch.flatten(t0.cuda()))
        # t0_dist = torch.gather(torch.flatten(t0.cuda()), 0, t0_indices)
        # t1_dist = torch.gather(torch.flatten(t1.cuda()), 0, t0_indices)
        # t0_dft = torch.fft.rfftn(t0_dist.float())
        # t1_dft = torch.fft.rfftn(t1_dist.float())
        # t1_filter = (.75 < (torch.arange(0, torch.numel(t0_dft), device='cuda') / (torch.numel(t0_dft) - 1))).float()
        # tx_dft = t0_dft + t1_filter * t1_dft
        # tx_dist = torch.fft.irfftn(tx_dft, t0_dist.size())
        # tx_values = torch.gather(tx_dist, 0, torch.argsort(t0_indices))
        # return tx_values.reshape(t0.shape).cpu()

        # t2_indices = torch.argsort(torch.flatten(((1 - alpha) * t0 + alpha * t1).cuda()))
        # t0_dist = torch.gather(torch.flatten(t0.cuda()), 0, t2_indices)
        # t1_dist = torch.gather(torch.flatten(t1.cuda()), 0, t2_indices)
        # t0_dft = torch.fft.rfftn(t0_dist.float())
        # t1_dft = torch.fft.rfftn(t1_dist.float())
        # dft_filter = (alpha < (torch.arange(0, torch.numel(t0_dft), device='cuda') / (torch.numel(t0_dft) - 1))).float()
        # tx_dft = dft_filter * t0_dft + (1 - dft_filter) * t1_dft
        # tx_dist = torch.fft.irfftn(tx_dft, t0_dist.size())
        # tx_values = torch.gather(tx_dist, 0, torch.argsort(t2_indices))
        # return tx_values.reshape(t0.shape).cpu()

        t2_dist, t2_indices = torch.sort(torch.flatten(t2.cuda()))
        t0_dist = torch.gather(torch.flatten(t0.cuda()), 0, t2_indices)
        t1_dist = torch.gather(torch.flatten(t1.cuda()), 0, t2_indices)
        #t0_dft = torch.fft.rfftn(t0_dist.float())
        #t1_dft = torch.fft.rfftn(t1_dist.float())

        dft_filter = (t0_dist + torch.min(t0_dist)) / (torch.max(t0_dist) - torch.min(t0_dist))

        # dft_filter = (alpha < (torch.arange(0, torch.numel(t0_dft), device='cuda') / (torch.numel(t0_dft) - 1))).float()
        # numel = int(2 * torch.numel(t0_dft) * 0.25)
        # dft_filter[int(alpha * torch.numel(t0_dft) - numel/2):int(alpha * torch.numel(t0_dft) + numel/2)] = \
        #      torch.Tensor([(math.cos(torch.pi * (1 - i/numel)) + 1)/2 for i in range(numel)])

        #tx_dft = (1 - dft_filter) * t0_dft + dft_filter * t1_dft
        #tx_dist = torch.fft.irfftn(tx_dft, t0_dist.size())

        tx_dist = (1 - dft_filter) * t0_dist + dft_filter * t1_dist

        import matplotlib.pyplot as plt
        # plt.plot(torch.arange(0, torch.numel(dft_filter)), dft_filter.cpu())
        # plt.show()

        tx_values = torch.gather(tx_dist, 0, torch.argsort(t2_indices))
        return tx_values.reshape(t0.shape).cpu()
    beta = current_bases["beta"]
    if merge_mode == "sum_twice":
        return (1 - beta) * ((1 - alpha) * t0 + alpha * t1) + beta * t2
    elif merge_mode == "triple_sum":
        return (1 - alpha - beta) * t0 + alpha * t1 + beta * t2


def save_model(model, output_file, file_format) -> None:
    if file_format == "safetensors":
        safetensors.torch.save_file(
            model, f"{output_file}.safetensors", metadata={"format": "pt"}
        )
    else:
        torch.save({"state_dict": model}, f"{output_file}.ckpt")
