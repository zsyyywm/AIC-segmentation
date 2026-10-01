"""AIC P v1: label-bound EMA prototypes and image-specific region context.

Pure PyTorch, with no MMSeg imports. Forward NEVER accepts labels. The head
calls update_from_labels once, after its prediction/loss, in training only.
This is a project-specific design, not a reproduction of ProtoSeg.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


PROTOTYPE_CONTRACT_VERSION = 'aic-p-ema-v1'


class MultiPrototypeRegionContext(nn.Module):
    """Same-shape residual context on B x 256 x H x W features.

    A class-labelled bank provides region affinities; weighted current-image
    features provide the region VALUES. No image is required to contain every
    class. Uninitialised slots receive zero affinity, not artificial features.
    Each update uses only pixels carrying that slot's official training class.
    """

    def __init__(self, feature_dim=256, num_classes=8, prototypes_per_class=4,
                 momentum=0.99, temperature=0.1, gate_init=1e-3,
                 max_update_pixels_per_class=2048, enabled=True,
                 ignore_index=255, eps=1e-6):
        super().__init__()
        if feature_dim != 256 or num_classes != 8 or prototypes_per_class != 4:
            raise ValueError('P v1 freezes feature_dim=256, classes=8, K=4')
        if not 0 <= momentum < 1 or not temperature > 0 or not eps > 0:
            raise ValueError('Invalid momentum, temperature or eps')
        if not all(math.isfinite(v) for v in
                   (momentum, temperature, gate_init, eps)):
            raise ValueError('P hyperparameters must be finite')
        if int(max_update_pixels_per_class) != max_update_pixels_per_class or max_update_pixels_per_class < 4:
            raise ValueError('Update cap must be an integer >= 4')
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.prototypes_per_class = prototypes_per_class
        self.momentum = float(momentum)
        self.temperature = float(temperature)
        self.max_update_pixels_per_class = int(max_update_pixels_per_class)
        self.ignore_index = ignore_index
        self.eps = float(eps)
        self.enabled = bool(enabled)
        self.query_proj = nn.Conv2d(feature_dim, feature_dim, 1, bias=False)
        self.value_proj = nn.Conv2d(feature_dim, feature_dim, 1, bias=False)
        self.context_proj = nn.Conv2d(feature_dim, feature_dim, 1, bias=False)
        self.residual_gate = nn.Parameter(torch.full((feature_dim,), gate_init))
        self.register_buffer('prototype_bank', torch.zeros(8, 4, feature_dim))
        self.register_buffer('prototype_valid', torch.zeros(8, 4, dtype=torch.bool))
        self.register_buffer('assignment_counts', torch.zeros(8, 4, dtype=torch.long))
        self.register_buffer('class_seen_pixels', torch.zeros(8, dtype=torch.long))
        self.register_buffer('update_steps', torch.zeros((), dtype=torch.long))
        # Transient, detached diagnostics; not checkpoint state or predictions.
        self._last = {}

    def _check_features(self, features):
        if features.ndim != 4 or features.shape[1] != self.feature_dim:
            raise ValueError('Expected B x 256 x H x W features')
        if not features.is_floating_point():
            raise TypeError('Features must be floating point')

    @staticmethod
    def _mask(features, valid_mask):
        if valid_mask is None:
            return torch.ones((features.shape[0], 1, *features.shape[-2:]),
                              device=features.device, dtype=torch.bool)
        if valid_mask.ndim == 3:
            valid_mask = valid_mask[:, None]
        if valid_mask.shape != (features.shape[0], 1, *features.shape[-2:]):
            raise ValueError('valid_mask must be B x 1 x H x W or B x H x W')
        if valid_mask.dtype != torch.bool:
            raise TypeError('valid_mask must be bool, derived from geometry')
        return valid_mask.to(device=features.device)

    def forward(self, features, valid_mask=None):
        self._check_features(features)
        if not self.enabled:
            self._last = {}
            return features
        valid = self._mask(features, valid_mask)
        if not self.prototype_valid.any() or not valid.any():
            self._last = {}
            return features
        batch, _, height, width = features.shape
        query = self.query_proj(features).flatten(2).transpose(1, 2)
        value = self.value_proj(features).flatten(2).transpose(1, 2)
        # loss() may update the buffers before backward. An immutable snapshot
        # is essential even though these buffers themselves require no grad.
        bank = self.prototype_bank.detach().clone().flatten(0, 1).float()
        slots = self.prototype_valid.detach().clone().flatten()
        with torch.autocast(device_type=features.device.type, enabled=False):
            query = F.normalize(query.float(), dim=-1, eps=self.eps)
            bank = F.normalize(bank, dim=-1, eps=self.eps)
            logits = torch.matmul(query, bank.T) / self.temperature
            logits = logits.masked_fill(~slots[None, None, :], float('-inf'))
            affinity = logits.softmax(dim=-1)
            affinity = affinity * valid.flatten(2).transpose(1, 2)
            support = affinity.sum(dim=1)  # B x 32, not normalised per class
            regions = torch.bmm(affinity.transpose(1, 2), value.float())
            regions = regions / support.clamp_min(self.eps)[..., None]
            context = torch.bmm(affinity, regions)
        context = context.transpose(1, 2).reshape(batch, self.feature_dim, height, width)
        delta = self.context_proj(context.to(features.dtype))
        delta = delta * self.residual_gate[None, :, None, None]
        delta = delta * valid
        with torch.no_grad():
            denominator = valid.sum().clamp_min(1)
            winners = affinity.detach().argmax(dim=-1)[valid.flatten(1)]
            self._last = {
                'attention_mass': (support.detach().sum(0) / denominator).reshape(8, 4),
                'winner_counts': torch.bincount(winners, minlength=32).reshape(8, 4),
                'residual_ratio': (delta.detach().float().norm() /
                                   features.detach().float().norm().clamp_min(self.eps)),
            }
        # Keep the small residual in fp32: MP/UP recover output-input as delta.
        # fp16 addition followed by subtraction can erase a 1e-3-gated branch.
        return features.float() + delta.float()

    @torch.no_grad()
    def update_from_labels(self, features, labels, valid_mask=None):
        """Public training-only update; EMA arithmetic stays float32 in AMP."""
        if not self.training or not self.enabled:
            return False
        self._check_features(features)
        with torch.autocast(device_type=features.device.type, enabled=False):
            return self._update_from_labels_float(features.detach().float(), labels, valid_mask)

    @torch.no_grad()
    def _update_from_labels_float(self, features, labels, valid_mask=None):
        """Commit training statistics AFTER the current prediction and loss.

        Returns True iff usable pixels were observed. Eval/disabled calls do
        nothing. Single-GPU v1 fails explicitly on multi-rank training rather
        than silently creating different banks on different ranks.
        """
        if not self.training or not self.enabled:
            return False
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            raise RuntimeError('P v1 statistics support single-GPU training only')
        self._check_features(features)
        valid = self._mask(features, valid_mask).squeeze(1)
        if labels.ndim == 4 and labels.shape[1] == 1:
            labels = labels[:, 0]
        if labels.ndim != 3 or labels.shape[0] != features.shape[0]:
            raise ValueError('Expected internal training labels B x Hgt x Wgt')
        if labels.is_floating_point():
            raise TypeError('Training labels must be integer IDs')
        labels = labels.to(features.device)
        legal = (labels == self.ignore_index) | ((labels >= 0) & (labels < 8))
        if not legal.all():
            raise ValueError('Labels must be internal 0..7 or Ignore255')
        if labels.shape[-2] < features.shape[-2] or labels.shape[-1] < features.shape[-1]:
            raise ValueError('GT resolution must be >= feature resolution')
        size = features.shape[-2:]
        low_labels = F.interpolate(labels[:, None].float(), size=size,
                                   mode='nearest')[:, 0].long()
        # Any Ignore in a feature cell excludes that cell from bank updates.
        # Nearest labels assign representative class IDs, never averaged IDs.
        ignored_cells = F.adaptive_max_pool2d(
            (labels == self.ignore_index)[:, None].float(), size)[:, 0].bool()
        valid = valid & ~ignored_cells
        query = self.query_proj(features.detach()).float()
        query = F.normalize(query, dim=1, eps=self.eps).permute(0, 2, 3, 1)
        valid = valid & torch.isfinite(query).all(dim=-1) & (query.norm(dim=-1) > self.eps)
        if not valid.any():
            return False
        for cls in range(8):
            selected = query[valid & (low_labels == cls)]
            total = selected.shape[0]
            if not total:
                continue  # absent classes keep their exact old state
            self.class_seen_pixels[cls] += total
            if total > self.max_update_pixels_per_class:
                indices = torch.linspace(0, total - 1, self.max_update_pixels_per_class,
                                         device=features.device).long()
                selected = selected[indices]
            # Deterministic farthest-point initialisation from actual labelled
            # features. Constant/insufficiently diverse data leaves slots cold.
            for slot in range(4):
                if self.prototype_valid[cls, slot]:
                    continue
                occupied = self.prototype_bank[cls, self.prototype_valid[cls]]
                if not len(occupied):
                    mean = F.normalize(selected.mean(0), dim=0, eps=self.eps)
                    scores = 1 - selected @ mean
                else:
                    scores = 1 - (selected @ occupied.T).max(dim=1).values
                    if scores.max() <= self.eps:
                        break
                chosen = selected[scores.argmax()]
                self.prototype_bank[cls, slot].copy_(chosen)
                self.prototype_valid[cls, slot] = True
            live_slots = self.prototype_valid[cls].nonzero(as_tuple=False).flatten()
            similarity = selected @ self.prototype_bank[cls, live_slots].T
            assigned = similarity.argmax(dim=1)
            for index, slot in enumerate(live_slots):
                members = selected[assigned == index]
                if not len(members):
                    continue
                mean = F.normalize(members.mean(0), dim=0, eps=self.eps)
                mixed = (self.momentum * self.prototype_bank[cls, slot]
                         + (1 - self.momentum) * mean)
                self.prototype_bank[cls, slot].copy_(
                    F.normalize(mixed, dim=0, eps=self.eps))
                self.assignment_counts[cls, slot] += len(members)
        self.update_steps += 1
        return True

    @torch.no_grad()
    def diagnostics(self):
        """JSON-ready read-only statistics; no features/GT are retained."""
        similarities = []
        for cls in range(8):
            bank = self.prototype_bank[cls, self.prototype_valid[cls]].float()
            if len(bank) < 2:
                similarities.append(None)
            else:
                cosine = F.normalize(bank, dim=-1) @ F.normalize(bank, dim=-1).T
                upper = torch.triu(torch.ones_like(cosine, dtype=torch.bool), diagonal=1)
                similarities.append(float(cosine[upper].max().cpu()))
        result = {
            'contract_version': PROTOTYPE_CONTRACT_VERSION,
            'enabled': self.enabled,
            'update_steps': int(self.update_steps.cpu()),
            'valid_slots': self.prototype_valid.cpu().tolist(),
            'assignment_counts': self.assignment_counts.cpu().tolist(),
            'class_seen_pixels': self.class_seen_pixels.cpu().tolist(),
            'max_within_class_cosine': similarities,
            'gate_mean_abs': float(self.residual_gate.detach().abs().mean().cpu()),
        }
        for name, value in self._last.items():
            result[name] = value.cpu().tolist()
        return result
