import torch
import json
import numpy

from ..builder import build_bbox_coder
from mmdet.core.bbox.iou_calculators import build_iou_calculator
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner
from ..builder import ROTATED_BBOX_ASSIGNERS
#from mmcv.utils import build_from_cfg


@ROTATED_BBOX_ASSIGNERS.register_module()
class C2FAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
        match_low_quality (bool): Whether to allow low quality matches. This is
            usually allowed for RPN and single stage detectors, but not allowed
            in the second stage. Details are demonstrated in Step 4.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
    """

    def __init__(self,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 gpu_assign_thr=512,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 assign_metric='gjsd',
                 topk=1,
                 topq=1,
                 constraint=False,
                 gauss_thr = 1.0,
                 bbox_coder=dict(
                     type='DeltaXYWHAOBBoxCoder',
                     target_means=(.0, .0, .0, .0, .0),
                     target_stds=(1.0, 1.0, 1.0, 1.0, 1.0))):
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.assign_metric = assign_metric
        self.topk = topk
        self.topq = topq
        self.constraint = constraint
        self.gauss_thr = gauss_thr
        self.bbox_coder = build_bbox_coder(bbox_coder)

    def assign(self, cls_scores, bbox_preds, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.
        """
        assign_on_cpu = True if (self.gpu_assign_thr >= 0) and (
            gt_bboxes.shape[0] > self.gpu_assign_thr) else False
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = bboxes.device
            bboxes = bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()

        overlaps = self.iou_calculator(gt_bboxes, bboxes, mode=self.assign_metric)

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(
                    bboxes, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(
                    gt_bboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes,),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes,),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, _ = overlaps.max(dim=0)
        # for each gt, topk anchors
        # for each gt, the topk of all proposals
        gt_max_overlaps, _ = overlaps.topk(self.topk, dim=1, largest=True, sorted=True)  # gt_argmax_overlaps [num_gt, k]
        assigned_gt_inds[(max_overlaps >= 0) & (max_overlaps < 0.8)] = 0

        for i in range(num_gts):
            for j in range(self.topk):
                max_overlap_inds = overlaps[i,:] == gt_max_overlaps[i,j]
                assigned_gt_inds[max_overlap_inds] = i + 1

        device = bboxes.device
        bbox_preds = bbox_preds.to(device)
        cls_scores = cls_scores.to(device)
        bbox_preds = torch.transpose(bbox_preds, 0, 1)
        bbox_preds = self.bbox_coder.decode(bboxes, bbox_preds)
        
        num_gt = gt_bboxes.size(0)
        num_bboxes = bboxes.size(0)

        can_positive_mask = assigned_gt_inds > 0
        can_positive_inds = torch.nonzero(can_positive_mask)

        poscan = assigned_gt_inds[can_positive_inds].squeeze(-1)
        can_other_mask = assigned_gt_inds <= 0

        can_pos_scores = cls_scores[:,can_positive_inds].squeeze(-1)

        can_pos_scores = torch.transpose(can_pos_scores, 0, 1)
        can_bbox_pred = bbox_preds[can_positive_inds,:].squeeze(-1)

        can_pos_iou = self.iou_calculator(gt_bboxes.to(device), can_bbox_pred, mode ='iou')
        can_pos_iou = can_pos_iou[poscan-1,range(poscan.size(0))]
        can_pos_cls, _ = torch.max(can_pos_scores,1)

        can_pos_quality = can_pos_iou + can_pos_cls.sigmoid() 
        can_pos_quality = can_pos_quality.unsqueeze(0).repeat(num_gt, 1) # size of gt, pos anchors
        
        gt_poscan = torch.zeros_like(can_pos_quality) - 100 # size of gt, pos anchors
        gt_poscan[poscan-1,range(poscan.size(0))] = can_pos_quality[poscan-1,range(poscan.size(0))]

        if self.topq >= can_pos_quality.size(1):
            topq = can_pos_quality.size(1)
        else:
            topq = self.topq
        gt_max_quality, gt_argmax_quality = gt_poscan.topk(topq, dim=1, largest=True, sorted=True)  # gt_argmax_quality [num_gt, q]

        assign_result_pre_gt = assigned_gt_inds

        assigned_gt_inds_init = assign_result_pre_gt * can_other_mask
        assigned_pos_prior = torch.zeros((num_gt, topq, 5),device=device)
        
        for i in range(num_gt):
            for j in range(topq):
                index = gt_argmax_quality[i,j]
                remap_inds = can_positive_inds[index,0]
                assigned_gt_inds_init[remap_inds] = assign_result_pre_gt [remap_inds]
                assigned_pos_prior[i,j,:] = bboxes[remap_inds,:] 
        assigned_gt_inds = assigned_gt_inds_init

        if self.constraint == 'dgmm':
            device1 = gt_bboxes.device
            xy_gt, sigma_t = self.xy_wh_r_2_xy_sigma(gt_bboxes)
            # get the mean of the positive samples
            pos_prior_mean = torch.mean(assigned_pos_prior[...,:2], dim=-2)
            _, sigma_t = self.xy_wh_r_2_xy_sigma(gt_bboxes)
            xy_pt = pos_prior_mean
            xy_a = bboxes[...,:2]
            xy_gt = xy_gt[...,None,:,:2].unsqueeze(-1)
            xy_pt = xy_pt[...,None,:,:2].unsqueeze(-1)
            xy_a = xy_a[...,:,None,:2].unsqueeze(-1)
            inv_sigma_t = torch.stack((sigma_t[..., 1, 1], -sigma_t[..., 0, 1],
                                      -sigma_t[..., 1, 0], sigma_t[..., 0, 0]),
                                      dim=-1).reshape(-1, 2, 2)
            inv_sigma_t = inv_sigma_t / sigma_t.det().unsqueeze(-1).unsqueeze(-1)
            gaussian_gt = torch.exp(-0.5*(xy_a-xy_gt).permute(0, 1, 3, 2).matmul(inv_sigma_t).matmul(xy_a-xy_gt)).squeeze(-1).squeeze(-1)
            gaussian_pt = torch.exp(-0.5*(xy_a-xy_pt).permute(0, 1, 3, 2).matmul(inv_sigma_t).matmul(xy_a-xy_pt)).squeeze(-1).squeeze(-1)
            gaussian = 0.7*gaussian_gt + 0.3*gaussian_pt 

            inside_flag = gaussian >= torch.exp(torch.tensor([-self.gauss_thr])).to(device1)
            length = range(assigned_gt_inds.size(0))
            inside_mask = inside_flag[length, (assigned_gt_inds-1).clamp(min=0)]
            assigned_gt_inds *= inside_mask

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        assign_result = AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        
        return assign_result

    def assign_wrt_ranking(self,  overlaps, gt_labels=None):
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes,),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes,),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, _ = overlaps.max(dim=0)
        # for each gt, topk anchors
        # for each gt, the topk of all proposals
        gt_max_overlaps, _ = overlaps.topk(self.topk, dim=1, largest=True, sorted=True)  # gt_argmax_overlaps [num_gt, k]


        assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < 0.8)] = 0

        for i in range(num_gts):
            for j in range(self.topk):
                max_overlap_inds = overlaps[i,:] == gt_max_overlaps[i,j]
                assigned_gt_inds[max_overlap_inds] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def xy_wh_r_2_xy_sigma(self, xywhr):
        """Convert oriented bounding box to 2-D Gaussian distribution.

        Args:
            xywhr (torch.Tensor): rbboxes with shape (N, 5).

        Returns:
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).
        """
        _shape = xywhr.shape
        assert _shape[-1] == 5
        xy = xywhr[..., :2]
        wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
        r = xywhr[..., 4]
        cos_r = torch.cos(r)
        sin_r = torch.sin(r)
        R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
        S = 0.5 * torch.diag_embed(wh)

        sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                                1)).reshape(_shape[:-1] + (2, 2))

        return xy, sigma



    