# Forward adapted from MMDetection 3.3.0 Mask2FormerHead (Apache-2.0).
# Copyright (c) OpenMMLab. All rights reserved.
"""Native Mask2Former with explicit valid-pixel supervision for AIC."""

import torch
from mmcv.ops import point_sample
from mmengine.structures import InstanceData
from mmdet.models.task_modules.assigners import AssignResult
from mmdet.models.utils import get_uncertain_point_coords_with_randomness
from mmdet.utils import reduce_mean
from mmseg.models.decode_heads.mask2former_head import Mask2FormerHead
from mmseg.registry import MODELS


@MODELS.register_module()
class AICMask2FormerHead(Mask2FormerHead):
    """Exclude Ignore and padding from matching, BCE, Dice and class targets.

    Full-valid images use native continuous point sampling. Images with Ignore
    use valid GT pixel centers, avoiding interpolation through invalid labels.
    Entirely ignored images contribute differentiable zero to ALL losses.
    The pixel/query decoders and their ten supervised outputs are unchanged.
    """

    def _seg_data_to_instance_data(self, batch_data_samples):
        batch_gt, batch_metas = [], []
        for sample in batch_data_samples:
            meta = dict(sample.metainfo)
            seg = sample.gt_sem_seg.data.squeeze(0)
            valid = seg != self.ignore_index
            img_h, img_w = meta.get('img_shape', seg.shape)[:2]
            spatial_valid = torch.zeros_like(valid)
            spatial_valid[:img_h, :img_w] = True
            valid = valid & spatial_valid
            labels = torch.unique(seg[valid]).long()
            if labels.numel() and ((labels < 0).any() or
                                   (labels >= self.num_classes).any()):
                raise ValueError('Expected internal classes 0..7 and Ignore255.')
            masks = (seg.unsqueeze(0) == labels[:, None, None]) & valid
            gt = InstanceData(labels=labels, masks=masks.long())
            # Image-level tensors belong in metainfo, not length-checked fields.
            gt.set_metainfo(dict(aic_valid_mask=valid))
            batch_gt.append(gt)
            batch_metas.append(meta)
        return batch_gt, batch_metas

    @staticmethod
    def _valid_coords(valid, batches, points):
        """Uniform samples with replacement from valid GT pixel centers."""
        yx = torch.nonzero(valid, as_tuple=False)
        if yx.shape[0] == 0:
            raise ValueError('Cannot sample an entirely ignored image.')
        indices = torch.randint(yx.shape[0], (batches, points), device=valid.device)
        sampled = yx[indices]
        height, width = valid.shape
        return torch.stack(((sampled[..., 1].float() + .5) / width,
                            (sampled[..., 0].float() + .5) / height), dim=-1)

    def _get_targets_single(self, cls_score, mask_pred, gt_instances, img_meta):
        valid = gt_instances.metainfo['aic_valid_mask']
        if bool(valid.all()):
            return super()._get_targets_single(
                cls_score, mask_pred, gt_instances, img_meta)
        num_queries = cls_score.shape[0]
        gt_labels, gt_masks = gt_instances.labels, gt_instances.masks
        if not bool(valid.any()):
            assign = AssignResult(
                num_gts=0,
                gt_inds=gt_labels.new_zeros(num_queries),
                max_overlaps=None,
                labels=gt_labels.new_full((num_queries,), -1))
        else:
            coords = self._valid_coords(valid, 1, self.num_points)
            predicted_points = point_sample(
                mask_pred.unsqueeze(1), coords.expand(num_queries, -1, -1)
            ).squeeze(1)
            target_points = point_sample(
                gt_masks.unsqueeze(1).float(),
                coords.expand(gt_labels.numel(), -1, -1)).squeeze(1)
            assign = self.assigner.assign(
                pred_instances=InstanceData(scores=cls_score,
                                            masks=predicted_points),
                gt_instances=InstanceData(labels=gt_labels, masks=target_points),
                img_meta=img_meta)
        sampling = self.sampler.sample(
            assign_result=assign,
            pred_instances=InstanceData(scores=cls_score, masks=mask_pred),
            gt_instances=gt_instances)
        if not bool(valid.any()):
            # Native MaskSamplingResult clamps num_pos to one. An ignored
            # image must not dilute valid images' BCE/Dice normalization.
            sampling.avg_factor = 0
        positive, negative = sampling.pos_inds, sampling.neg_inds
        labels = gt_labels.new_full((num_queries,), self.num_classes)
        labels[positive] = gt_labels[sampling.pos_assigned_gt_inds]
        label_weights = cls_score.new_full((num_queries,), float(valid.any()))
        mask_weights = mask_pred.new_zeros(num_queries)
        mask_weights[positive] = 1.
        mask_targets = gt_masks[sampling.pos_assigned_gt_inds]
        return (labels, label_weights, mask_targets, mask_weights,
                positive, negative, sampling)

    def _valid_uncertain_coords(self, mask_predictions, valid):
        count = mask_predictions.shape[0]
        candidates = self._valid_coords(
            valid, count, int(self.num_points * self.oversample_ratio))
        uncertainty = -point_sample(
            mask_predictions.unsqueeze(1), candidates).squeeze(1).abs()
        important = int(self.num_points * self.importance_sample_ratio)
        indices = uncertainty.topk(important, dim=1).indices
        selected = candidates.gather(1, indices[..., None].expand(-1, -1, 2))
        random = self._valid_coords(valid, count, self.num_points - important)
        return torch.cat((selected, random), dim=1)

    def _loss_by_feat_single(self, cls_scores, mask_preds,
                             batch_gt_instances, batch_img_metas):
        # Matching and point losses remain FP32 under the AMP training wrapper.
        with torch.autocast(device_type=mask_preds.device.type, enabled=False):
            return self._loss_by_feat_single_fp32(
                cls_scores.float(), mask_preds.float(),
                batch_gt_instances, batch_img_metas)

    def _loss_by_feat_single_fp32(self, cls_scores, mask_preds,
                                  batch_gt_instances, batch_img_metas):
        if not any(bool(gt.metainfo['aic_valid_mask'].any())
                   for gt in batch_gt_instances):
            return cls_scores.sum() * 0., mask_preds.sum() * 0., mask_preds.sum() * 0.
        batch = cls_scores.shape[0]
        (labels_list, weights_list, targets_list, mask_weights_list,
         avg_factor) = self.get_targets(
            [cls_scores[i] for i in range(batch)],
            [mask_preds[i] for i in range(batch)],
            batch_gt_instances, batch_img_metas)
        labels = torch.stack(labels_list).flatten()
        weights = torch.stack(weights_list).flatten()
        class_weight = cls_scores.new_tensor(self.class_weight)
        normalizer = (class_weight[labels] * weights).sum().clamp_min(1.)
        loss_cls = self.loss_cls(cls_scores.flatten(0, 1), labels, weights,
                                 avg_factor=normalizer)
        num_masks = max(reduce_mean(cls_scores.new_tensor([avg_factor])), 1)
        targets = torch.cat(targets_list)
        positive_preds = mask_preds[torch.stack(mask_weights_list) > 0]
        if targets.shape[0] == 0:
            return loss_cls, positive_preds.sum(), positive_preds.sum()
        coords_parts, offset = [], 0
        with torch.no_grad():
            for gt, mask_weights in zip(batch_gt_instances, mask_weights_list):
                count = int((mask_weights > 0).sum())
                if not count:
                    continue
                predictions = positive_preds[offset:offset + count]
                valid = gt.metainfo['aic_valid_mask']
                if bool(valid.all()):
                    coords = get_uncertain_point_coords_with_randomness(
                        predictions.unsqueeze(1), None, self.num_points,
                        self.oversample_ratio, self.importance_sample_ratio)
                else:
                    coords = self._valid_uncertain_coords(predictions, valid)
                coords_parts.append(coords)
                offset += count
            coords = torch.cat(coords_parts)
            point_targets = point_sample(targets.unsqueeze(1).float(), coords).squeeze(1)
        point_predictions = point_sample(positive_preds.unsqueeze(1), coords).squeeze(1)
        loss_dice = self.loss_dice(point_predictions, point_targets,
                                   avg_factor=num_masks)
        loss_mask = self.loss_mask(point_predictions.flatten(), point_targets.flatten(),
                                   avg_factor=num_masks * self.num_points)
        return loss_cls, loss_mask, loss_dice

    def _process_pixel_decoder_outputs(self, mask_features,
                                       multi_scale_memorys, batch_data_samples):
        """MP extension point; M0 returns the identical tensor/list objects."""
        return mask_features, multi_scale_memorys

    def predict(self, x, batch_img_metas, test_cfg):
        # EncoderDecoder.slide_inference updates img_shape per window but can
        # leave pad_shape referring to the full image. The shared wrapper uses
        # pad_shape first; normalize it locally to the CURRENT slide window.
        if test_cfg.get('mode') == 'slide':
            batch_img_metas = [dict(meta, pad_shape=meta['img_shape'])
                               for meta in batch_img_metas]
        return super().predict(x, batch_img_metas, test_cfg)

    def forward(self, x, batch_data_samples):
        """MMDetection 3.3.0 forward, with one identity feature-processing hook."""
        batch_size = x[0].shape[0]
        mask_features, memories = self.pixel_decoder(x)
        mask_features, memories = self._process_pixel_decoder_outputs(
            mask_features, memories, batch_data_samples)
        inputs, positional_encodings = [], []
        for i in range(self.num_transformer_feat_level):
            memory = self.decoder_input_projs[i](memories[i])
            memory = memory.flatten(2).permute(0, 2, 1)
            memory = memory + self.level_embed.weight[i].view(1, 1, -1)
            padding = memory.new_zeros((batch_size,) + memories[i].shape[-2:],
                                       dtype=torch.bool)
            position = self.decoder_positional_encoding(padding)
            inputs.append(memory)
            positional_encodings.append(position.flatten(2).permute(0, 2, 1))
        query = self.query_feat.weight.unsqueeze(0).repeat((batch_size, 1, 1))
        query_position = self.query_embed.weight.unsqueeze(0).repeat((batch_size, 1, 1))
        classes, masks = [], []
        cls, mask, attention = self._forward_head(query, mask_features,
                                                 memories[0].shape[-2:])
        classes.append(cls)
        masks.append(mask)
        for i in range(self.num_transformer_decoder_layers):
            level = i % self.num_transformer_feat_level
            allowed = (attention.sum(-1) != attention.shape[-1]).unsqueeze(-1)
            attention = attention & allowed
            query = self.transformer_decoder.layers[i](
                query=query, key=inputs[level], value=inputs[level],
                query_pos=query_position, key_pos=positional_encodings[level],
                cross_attn_mask=attention, query_key_padding_mask=None,
                key_padding_mask=None)
            cls, mask, attention = self._forward_head(
                query, mask_features, memories[(i + 1) % self.num_transformer_feat_level].shape[-2:])
            classes.append(cls)
            masks.append(mask)
        return classes, masks
