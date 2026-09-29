"""Pure-PyTorch RMI regional term, independently implemented from the paper.

Zhao et al., NeurIPS 2019, equations (12), (14), (16):
https://proceedings.neurips.cc/paper/2019/file/a67c8c9a961b4182688768dd9ba015fe-Paper.pdf

This is NOT the complete CE + RMI objective: CE is configured separately.
The minimized term is +logdet(conditional covariance)/(2 * region dimension).
It may be negative; adding a constant or clamping it to zero is not appropriate.
"""

import torch
from torch import Tensor
from torch.nn import functional as F


def regional_mutual_information(
    logits: Tensor,
    target: Tensor,
    *,
    num_classes: int = 8,
    radius: int = 3,
    pool_size: int = 4,
    covariance_epsilon: float = 1e-6,
    ignore_index: int = 255,
) -> Tensor:
    """Return class-summed, batch-averaged RMI, without a CE or loss weight.

    RMI uses sigmoid probabilities per the original paper. Average pooling
    reduces memory before extracting sliding radius x radius neighborhoods.
    A neighborhood is usable only if EVERY contributing original pixel is
    valid. This deliberately stricter Ignore treatment avoids learning from
    artificial zero-valued labels at annotation/padding boundaries.

    Covariances use the population normalization 1/N and float64 arithmetic
    with autocast disabled. Cholesky solves replace an explicit inverse.
    Too-small maps or fewer than two valid neighborhoods contribute a
    differentiable zero. An ignored image remains in the batch denominator.
    """
    if logits.ndim != 4 or target.ndim != 3:
        raise ValueError('Expected logits NCHW and target NHW')
    if logits.shape[0] == 0 or logits.shape[2] == 0 or logits.shape[3] == 0:
        raise ValueError('Empty batch or spatial dimension')
    if logits.shape[1] != num_classes or target.shape != (
        logits.shape[0], logits.shape[2], logits.shape[3]
    ):
        raise ValueError('Target shape or num_classes does not match logits')
    if target.device != logits.device:
        raise ValueError('Logits and target must be on the same device')
    if target.dtype not in (torch.uint8, torch.int8, torch.int16,
                            torch.int32, torch.int64):
        raise TypeError('Target must contain integer class IDs')
    if not logits.is_floating_point():
        raise TypeError('Logits must be floating point')
    if radius < 1 or pool_size < 1 or int(radius) != radius or int(pool_size) != pool_size:
        raise ValueError('radius and pool_size must be positive integers')
    radius, pool_size = int(radius), int(pool_size)
    if not (0 < covariance_epsilon < float('inf')):
        raise ValueError('covariance_epsilon must be finite and positive')
    valid = target != ignore_index
    if bool(((target[valid] < 0) | (target[valid] >= num_classes)).any()):
        raise ValueError('Non-ignore targets must be in [0, num_classes)')
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError('Non-finite logits passed to RMI')

    # Explicit fp32 probability computation prevents AMP sigmoid underflow;
    # double logits are preserved for numerical reference / gradcheck tests.
    with torch.autocast(device_type=logits.device.type, enabled=False):
        work = logits if logits.dtype == torch.float64 else logits.float()
        zero = work.sum() * 0.0
        if min(logits.shape[-2:]) < pool_size * radius or not bool(valid.any()):
            return zero
        prob = work.sigmoid()
        labels = F.one_hot(target.masked_fill(~valid, 0).long(), num_classes)
        labels = labels.permute(0, 3, 1, 2).to(work.dtype)
        labels = labels * valid[:, None]
        prob = prob.masked_fill(~valid[:, None], 0)
        if pool_size > 1:
            labels = F.avg_pool2d(labels, pool_size, pool_size)
            prob = F.avg_pool2d(prob, pool_size, pool_size)
            # No padding; an incomplete right/bottom pooling tile is omitted.
            valid = F.max_pool2d((~valid[:, None]).float(),
                                 pool_size, pool_size)[:, 0] == 0
        dim = radius * radius
        eye = torch.eye(dim, dtype=torch.float64, device=logits.device)
        total = zero.double()
        for image in range(logits.shape[0]):
            windows = F.unfold(valid[image:image + 1, None].float(), radius)
            keep = windows[0].amin(dim=0) == 1
            count = int(keep.sum())
            if count < 2:
                continue
            y = F.unfold(labels[image:image + 1], radius)
            p = F.unfold(prob[image:image + 1], radius)
            y = y.reshape(num_classes, dim, -1)[..., keep].double()
            p = p.reshape(num_classes, dim, -1)[..., keep].double()
            y = y - y.mean(dim=-1, keepdim=True)
            p = p - p.mean(dim=-1, keepdim=True)
            cy = (y @ y.transpose(-1, -2)) / count
            cp = (p @ p.transpose(-1, -2)) / count
            cyp = (y @ p.transpose(-1, -2)) / count
            chol_p = torch.linalg.cholesky(cp + covariance_epsilon * eye)
            solved = torch.cholesky_solve(cyp.transpose(-1, -2), chol_p)
            conditional = cy - cyp @ solved
            conditional = (conditional + conditional.transpose(-1, -2)) * 0.5
            chol_y = torch.linalg.cholesky(conditional + covariance_epsilon * eye)
            # logdet(M) = 2 sum(log(diag(cholesky(M)))).
            total = total + chol_y.diagonal(dim1=-2, dim2=-1).log().sum() / dim
        result = total / logits.shape[0]
        if not bool(torch.isfinite(result)):
            raise FloatingPointError('Non-finite RMI result; do not silently skip')
        return result.to(work.dtype)
