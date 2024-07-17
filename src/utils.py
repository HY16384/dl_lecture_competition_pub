import random
import numpy as np
import torch
from typing import Optional
from torch import Tensor

def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

import torch
import numpy as np
from enum import Enum, auto
from time import time


class RepresentationType(Enum):
    VOXEL = auto()
    STEPAN = auto()


class EventRepresentation:
    def __init__(self):
        pass

    def convert(self, events):
        raise NotImplementedError


class VoxelGrid(EventRepresentation):
    def __init__(self, input_size: tuple, normalize: bool):
        assert len(input_size) == 3
        self.voxel_grid = torch.zeros(
            (input_size), dtype=torch.float, requires_grad=False)
        self.nb_channels = input_size[0]
        self.normalize = normalize

    def convert(self, events):
        C, H, W = self.voxel_grid.shape
        with torch.no_grad():
            self.voxel_grid = self.voxel_grid.to(events['p'].device)
            voxel_grid = self.voxel_grid.clone()

            t_norm = events['t']
            t_norm = (C - 1) * (t_norm-t_norm[0]) / (t_norm[-1]-t_norm[0])

            x0 = events['x'].int()
            y0 = events['y'].int()
            t0 = t_norm.int()

            value = 2*events['p']-1
            #start_t = time()
            for xlim in [x0, x0+1]:
                for ylim in [y0, y0+1]:
                    for tlim in [t0, t0+1]:

                        mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (
                            ylim >= 0) & (tlim >= 0) & (tlim < self.nb_channels)
                        interp_weights = value * (1 - (xlim-events['x']).abs()) * (
                            1 - (ylim-events['y']).abs()) * (1 - (tlim - t_norm).abs())
                        index = H * W * tlim.long() + \
                            W * ylim.long() + \
                            xlim.long()

                        voxel_grid.put_(
                            index[mask], interp_weights[mask], accumulate=True)

            if self.normalize:
                mask = torch.nonzero(voxel_grid, as_tuple=True)
                if mask[0].size()[0] > 0:
                    mean = voxel_grid[mask].mean()
                    std = voxel_grid[mask].std()
                    if std > 0:
                        voxel_grid[mask] = (voxel_grid[mask] - mean) / std
                    else:
                        voxel_grid[mask] = voxel_grid[mask] - mean

        return voxel_grid


class PolarityCount(EventRepresentation):
    def __init__(self, input_size: tuple):
        assert len(input_size) == 3
        self.voxel_grid = torch.zeros(
            (input_size), dtype=torch.float, requires_grad=False)
        self.nb_channels = input_size[0]

    def convert(self, events):
        C, H, W = self.voxel_grid.shape
        with torch.no_grad():
            self.voxel_grid = self.voxel_grid.to(events['p'].device)
            voxel_grid = self.voxel_grid.clone()

            x0 = events['x'].int()
            y0 = events['y'].int()

            #start_t = time()
            for xlim in [x0, x0+1]:
                for ylim in [y0, y0+1]:
                    mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (
                        ylim >= 0)
                    interp_weights = (1 - (xlim-events['x']).abs()) * (
                        1 - (ylim-events['y']).abs())
                    index = H * W * events['p'].long() + \
                        W * ylim.long() + \
                        xlim.long()

                    voxel_grid.put_(
                        index[mask], interp_weights[mask], accumulate=True)

        return voxel_grid


def flow_16bit_to_float(flow_16bit: np.ndarray):
    assert flow_16bit.dtype == np.uint16
    assert flow_16bit.ndim == 3
    h, w, c = flow_16bit.shape
    assert c == 3

    valid2D = flow_16bit[..., 2] == 1
    assert valid2D.shape == (h, w)
    assert np.all(flow_16bit[~valid2D, -1] == 0)
    valid_map = np.where(valid2D)

    # to actually compute something useful:
    flow_16bit = flow_16bit.astype('float')

    flow_map = np.zeros((h, w, 2))
    flow_map[valid_map[0], valid_map[1], 0] = (
        flow_16bit[valid_map[0], valid_map[1], 0] - 2 ** 15) / 128
    flow_map[valid_map[0], valid_map[1], 1] = (
        flow_16bit[valid_map[0], valid_map[1], 1] - 2 ** 15) / 128
    return flow_map, valid2D


# https://github.com/pytorch/vision/blob/main/torchvision/models/optical_flow/_utils.py

def grid_sample(img: Tensor, absolute_grid: Tensor, mode: str = "bilinear", align_corners: Optional[bool] = None):
    """Same as torch's grid_sample, with absolute pixel coordinates instead of normalized coordinates."""
    h, w = img.shape[-2:]

    xgrid, ygrid = absolute_grid.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (w - 1) - 1
    # Adding condition if h > 1 to enable this function be reused in raft-stereo
    if h > 1:
        ygrid = 2 * ygrid / (h - 1) - 1
    normalized_grid = torch.cat([xgrid, ygrid], dim=-1)

    return F.grid_sample(img, normalized_grid, mode=mode, align_corners=align_corners)

def make_coords_grid(batch_size: int, h: int, w: int, device: str = "cpu"):
    device = torch.device(device)
    coords = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch_size, 1, 1, 1)

def upsample_flow(flow, up_mask: Optional[Tensor] = None, factor: int = 8):
    """Upsample flow by the input factor (default 8).

    If up_mask is None we just interpolate.
    If up_mask is specified, we upsample using a convex combination of its weights. See paper page 8 and appendix B.
    Note that in appendix B the picture assumes a downsample factor of 4 instead of 8.
    """
    batch_size, num_channels, h, w = flow.shape
    new_h, new_w = h * factor, w * factor

    if up_mask is None:
        return factor * F.interpolate(flow, size=(new_h, new_w), mode="bilinear", align_corners=True)

    up_mask = up_mask.view(batch_size, 1, 9, factor, factor, h, w)
    up_mask = torch.softmax(up_mask, dim=2)  # "convex" == weights sum to 1

    upsampled_flow = F.unfold(factor * flow, kernel_size=3, padding=1).view(batch_size, num_channels, 9, 1, 1, h, w)
    upsampled_flow = torch.sum(up_mask * upsampled_flow, dim=2)

    return upsampled_flow.permute(0, 1, 4, 2, 5, 3).reshape(batch_size, num_channels, new_h, new_w)