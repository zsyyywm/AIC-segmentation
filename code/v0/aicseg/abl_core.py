"""Independent AIC implementation of Wang et al., Active Boundary Loss (2022).

Algorithm reference: https://arxiv.org/abs/2102.02696 (equations 1--5).
Reference implementation reviewed: wangchi95/active-boundary-loss (Apache-2.0).
This implementation is written for the paper's algorithm, not a verbatim copy.
Project adaptations are documented in version_notes.md. No network is added.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from torch import nn


class ActiveBoundaryLossCore(nn.Module):
    """Sparse directional KL supervision with detached neighbor distributions.

    Expects full-resolution NCHW logits and NHW internal labels. The decoder
    already aligns logits. Ignore pixels and their 3x3 neighborhoods cannot
    contribute. Distances are computed on CPU; all differentiable work is FP32.
    """

    OFFSETS = ((1, 0), (-1, 0), (0, -1), (0, 1),
               (-1, 1), (1, 1), (-1, -1), (1, -1))

    def __init__(self, loss_weight=0.1, max_boundary_ratio=0.01,
                 min_boundary_kl=1e-5, max_distance=20.0,
                 label_smoothing=0.2, ignore_index=255,
                 loss_name='loss_abl'):
        super().__init__()
        if not 0 < max_boundary_ratio <= 1:
            raise ValueError('max_boundary_ratio must be in (0, 1]')
        if not 0 <= label_smoothing < 1:
            raise ValueError('label_smoothing must be in [0, 1)')
        if (max_distance <= 0 or min_boundary_kl < 0 or loss_weight < 0
                or not all(math.isfinite(x) for x in
                           (max_distance, min_boundary_kl, loss_weight))):
            raise ValueError('loss parameters must be finite and nonnegative')
        self.loss_weight = loss_weight
        self.max_boundary_ratio = max_boundary_ratio
        self.min_boundary_kl = min_boundary_kl
        self.max_distance = max_distance
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    @property
    def loss_name(self):
        return self._loss_name

    @staticmethod
    def geometry(labels, ignore_index):
        """CPU GT boundary distance and conservative valid center mask.

        Directed down/right label changes match the paper's two-neighborhood.
        Ignore never becomes a GT boundary. If an ignored region is nearer
        than the closest GT boundary, omit the center to avoid attraction
        across missing labels. Border centers are omitted (need 8 neighbors).
        """
        valid = labels != ignore_index
        edge = np.zeros(labels.shape, dtype=bool)
        edge[:-1] |= ((labels[:-1] != labels[1:])
                     & valid[:-1] & valid[1:])
        edge[:, :-1] |= ((labels[:, :-1] != labels[:, 1:])
                        & valid[:, :-1] & valid[:, 1:])
        safe = valid.copy()
        safe[[0, -1], :] = False
        safe[:, [0, -1]] = False
        for dy, dx in ActiveBoundaryLossCore.OFFSETS:
            safe[1:-1, 1:-1] &= valid[1+dy:labels.shape[0]-1+dy,
                                     1+dx:labels.shape[1]-1+dx]
        if not edge.any():
            return np.zeros(labels.shape, dtype=np.float32), safe & False
        distance = distance_transform_edt(~edge).astype(np.float32)
        if not valid.all():
            safe &= distance < distance_transform_edt(valid)
        return distance, safe

    @torch.no_grad()
    def predicted_boundaries(self, log_probability, safe):
        """Top KL changes per image, followed by one-pixel dilation.

        Top-k implements a bounded adaptive threshold without a Python
        epsilon-increment loop. Ties may pick any equivalent top-k pixels.
        """
        probability = log_probability.exp()
        score = torch.zeros_like(safe, dtype=torch.float32)
        score[:-1] += (probability[:, :-1] *
                      (log_probability[:, :-1] - log_probability[:, 1:])).sum(0)
        score[:, :-1] += (probability[:, :, :-1] *
                         (log_probability[:, :, :-1] -
                          log_probability[:, :, 1:])).sum(0)
        eligible = safe & (score > self.min_boundary_kl)
        values = score.masked_fill(~eligible, -torch.inf).flatten()
        count = max(1, math.floor(score.numel() * self.max_boundary_ratio))
        indices = values.topk(count, sorted=False).indices
        selected = torch.zeros_like(values, dtype=torch.bool)
        selected[indices] = torch.isfinite(values[indices])
        selected = F.max_pool2d(selected.reshape(1, 1, *score.shape).float(),
                                kernel_size=3, stride=1, padding=1)[0, 0] > 0
        return selected & safe

    def forward(self, pred, target, weight=None, ignore_index=None,
                reduction_override=None, **kwargs):
        if weight is not None:
            raise ValueError('ABL does not support pixel samplers/weights')
        if reduction_override not in (None, 'mean'):
            raise ValueError('ABL only supports mean reduction')
        if pred.ndim != 4 or target.ndim != 3:
            raise ValueError('ABL expects NCHW logits and NHW target')
        if pred.shape[0] != target.shape[0] or pred.shape[2:] != target.shape[1:]:
            raise ValueError('Align logits to target before calling ABL')
        if pred.device != target.device or pred.shape[1] < 2:
            raise ValueError('Need >=2 classes and logits/target on same device')
        if target.is_floating_point():
            raise ValueError('Target must contain integer class IDs')
        ignore = self.ignore_index if ignore_index is None else ignore_index
        labels_cpu = target.detach().cpu().numpy()
        valid = labels_cpu != ignore
        if np.any((labels_cpu[valid] < 0) | (labels_cpu[valid] >= pred.shape[1])):
            raise ValueError('Target contains out-of-range non-ignore IDs')
        # Disable outer autocast explicitly: KL and log-softmax require FP32.
        with torch.autocast(device_type=pred.device.type, enabled=False):
            logits = pred.float()
            zero = logits.sum() * 0.0
            if min(target.shape[-2:]) < 3 or not valid.any():
                return zero
            log_probability = logits.log_softmax(dim=1)
            loss_sum, count = zero, 0
            for batch_index, labels in enumerate(labels_cpu):
                distance_cpu, safe_cpu = self.geometry(labels, ignore)
                if not safe_cpu.any():
                    continue
                distance = torch.as_tensor(distance_cpu, device=pred.device)
                safe = torch.as_tensor(safe_cpu, device=pred.device)
                logp = log_probability[batch_index]
                centers = self.predicted_boundaries(logp.detach(), safe)
                y, x = torch.where(centers & (distance > 0))
                if y.numel() == 0:
                    continue
                neighboring_distances = torch.stack([
                    distance[y+dy, x+dx] for dy, dx in self.OFFSETS], dim=1)
                nearest, direction = neighboring_distances.min(dim=1)
                advances = nearest < distance[y, x]
                y, x, direction = y[advances], x[advances], direction[advances]
                if y.numel() == 0:
                    continue
                center_logp = logp[:, y, x].transpose(0, 1)
                neighbor_logp = torch.stack([
                    logp[:, y+dy, x+dx].transpose(0, 1).detach()
                    for dy, dx in self.OFFSETS], dim=1)
                # KL(neighbor || center); only center carries gradients.
                direction_logits = (neighbor_logp.exp() *
                                    (neighbor_logp - center_logp[:, None])).sum(2)
                direction_logp = direction_logits.log_softmax(dim=1)
                nll = -direction_logp.gather(1, direction[:, None]).squeeze(1)
                # Paper smoothing: target .8, each other direction .2/7.
                other_nll = (-direction_logp.sum(1) - nll) / 7.0
                per_center = ((1-self.label_smoothing) * nll
                              + self.label_smoothing * other_nll)
                distance_weight = distance[y, x].clamp(max=self.max_distance)
                loss_sum = loss_sum + (per_center * distance_weight /
                                       self.max_distance).sum()
                count += y.numel()
            return self.loss_weight * loss_sum / max(count, 1)
