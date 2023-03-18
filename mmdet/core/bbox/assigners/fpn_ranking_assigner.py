import torch

from ..builder import BBOX_ASSIGNERS, build_bbox_coder
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class FPNRankingAssigner(BaseAssigner):
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
                 assign_metric='iou',
                 topk=1,
                 topq=10,
                 cls_w = 1.0,
                 inside_circle=False,
                 gauss_thr = 1.5,
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
        self.cls_w = cls_w
        self.inside_circle = inside_circle
        self.gauss_thr = gauss_thr
        self.bbox_coder = build_bbox_coder(bbox_coder)

    def assign(self, cls_scores, bbox_preds, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, or a semi-positive number. -1 means negative
        sample, semi-positive number is the index (0-based) of assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to the background
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.

        Example:
            >>> self = MaxIoUAssigner(0.5, 0.5)
            >>> bboxes = torch.Tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
            >>> gt_bboxes = torch.Tensor([[0, 0, 10, 9]])
            >>> assign_result = self.assign(bboxes, gt_bboxes)
            >>> expected_gt_inds = torch.LongTensor([1, 0])
            >>> assert torch.all(assign_result.gt_inds == expected_gt_inds)
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

        #assign_result =self.lvl_assign_wrt_ranking(gt_bboxes, overlaps, gt_labels)

        assign_result =self.assign_wrt_ranking(overlaps, gt_labels)

        ######### POSTERIOR #############
        device = bboxes.device
        bbox_preds = bbox_preds.to(device)
        cls_scores = cls_scores.to(device)
        bbox_preds = torch.transpose(bbox_preds, 0, 1)

        bbox_preds = self.bbox_coder.decode(bboxes, bbox_preds)
        
        num_gt = gt_bboxes.size(0)
        num_bboxes = bboxes.size(0)

        # loss-aware assigning within deformable candidate box
        can_positive_mask = assign_result.gt_inds > 0
        can_positive_inds = torch.nonzero(can_positive_mask)
        #print(can_positive_inds)

        poscan = assign_result.gt_inds[can_positive_inds].squeeze(-1)
        can_other_mask = assign_result.gt_inds <= 0

        can_pos_scores = cls_scores[:,can_positive_inds].squeeze(-1)#.to(device)

        can_pos_scores = torch.transpose(can_pos_scores, 0, 1)
        can_bbox_pred = bbox_preds[can_positive_inds,:].squeeze()#.to(device)
        #can_bbox_pred = torch.transpose(can_bbox_pred, 0, 1)

        can_pos_iou = self.iou_calculator(gt_bboxes.to(device), can_bbox_pred, mode ='iou')
        can_pos_iou = can_pos_iou[poscan-1,range(poscan.size(0))]
        can_pos_cls, _ = torch.max(can_pos_scores,1)

        #self.count +=1

        #if self.count % 50 == 0:
        #    print(can_pos_cls)

        can_pos_quality = can_pos_iou + can_pos_cls.sigmoid() * self.cls_w
        can_pos_quality = can_pos_quality.unsqueeze(0).repeat(num_gt, 1) # size of gt, pos anchors
        
        gt_poscan = torch.zeros_like(can_pos_quality) - 100 # size of gt, pos anchors
        gt_poscan[poscan-1,range(poscan.size(0))] = can_pos_quality[poscan-1,range(poscan.size(0))]


        if self.topq >= can_pos_quality.size(1):
            topq = can_pos_quality.size(1)
        else:
            topq = self.topq
        gt_max_quality, gt_argmax_quality = gt_poscan.topk(topq, dim=1, largest=True, sorted=True)  # gt_argmax_quality [num_gt, q]

        assign_result_pre_gt = assign_result.gt_inds

        assigned_gt_inds_init = assign_result_pre_gt * can_other_mask
        assigned_pos_prior = torch.zeros((num_gt, topq, 5),device=device)
        #assigned_pos_prior_t1 = torch.zeros((num_gt, 1, 5), device=device)

        for i in range(num_gt):
            for j in range(topq):
                index = gt_argmax_quality[i,j]
                remap_inds = can_positive_inds[index,0]
                assigned_gt_inds_init[remap_inds] = assign_result_pre_gt[remap_inds]
                assigned_pos_prior[i,j,:] = bboxes[remap_inds,:] # get the location of assigned positive prior for each gt
                #print(assign_result.gt_inds[can_positive_inds[index,0]])
        assign_result.gt_inds = assigned_gt_inds_init




        ########## Fine-grained instance        

        if self.inside_circle == 'circle':
            center_distance = self.iou_calculator(gt_bboxes, bboxes, mode = 'center_distance2')
            width_gt = gt_bboxes[...,2]
            height_gt = gt_bboxes[...,3]
            # scale [0, 32]^2 r=2, scale [32, 256]^2 r= 1.5, scale [256, +inf]^2 r=1 for scale normalization
            '''
            scale = width_gt * height_gt
            scale_1 = scale <= 32*32
            scale_2 = (scale> 32*32) & (scale <= 256*256)
            scale_3 = scale > 256*256
            r = [2, 1.5, 1]
            gt_circle2 = ((width_gt/2)**2 + (height_gt/2) **2) *(scale_1*r[0]**2+scale_2*r[1]**2+scale_3*r[2]**2)
            '''
            r=1.5
            gt_circle = ((width_gt/2)**2 + (height_gt/2) **2) * r * r
            inside_flag = center_distance <= gt_circle[...,None]
            length = range(assign_result.gt_inds.size(0))
            inside_mask = inside_flag[(assign_result.gt_inds-1).clamp(min=0), length]
            assign_result.gt_inds *= inside_mask

        elif self.inside_circle == 'gaussian':
            device1 = gt_bboxes.device
            xy_t, sigma_t = self.xy_wh_r_2_xy_sigma(gt_bboxes)
            xy_a = bboxes[...,:2]
            xy_t = xy_t[...,None,:,:2].unsqueeze(-1)
            xy_a = xy_a[...,:,None,:2].unsqueeze(-1)
            inv_sigma_t = torch.stack((sigma_t[..., 1, 1], -sigma_t[..., 0, 1],
                                      -sigma_t[..., 1, 0], sigma_t[..., 0, 0]),
                                      dim=-1).reshape(-1, 2, 2)
            inv_sigma_t = inv_sigma_t / sigma_t.det().unsqueeze(-1).unsqueeze(-1)
            gaussian = torch.exp(-0.5*(xy_a-xy_t).permute(0, 1, 3, 2).matmul(inv_sigma_t).matmul(xy_a-xy_t)).squeeze(-1).squeeze(-1) #/(2*3.1415926*sigma_t.det())
            inside_flag = gaussian >= torch.exp(torch.tensor([-self.gauss_thr])).to(device1)
            length = range(assign_result.gt_inds.size(0))
            inside_mask = inside_flag[length, (assign_result.gt_inds-1).clamp(min=0)]
            assign_result.gt_inds *= inside_mask

        elif self.inside_circle == 'dgmm':
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
            gaussian_gt = torch.exp(-0.5*(xy_a-xy_gt).permute(0, 1, 3, 2).matmul(inv_sigma_t).matmul(xy_a-xy_gt)).squeeze(-1).squeeze(-1) #/(2*3.1415926*sigma_t.det())
            gaussian_pt = torch.exp(-0.5*(xy_a-xy_pt).permute(0, 1, 3, 2).matmul(inv_sigma_t).matmul(xy_a-xy_pt)).squeeze(-1).squeeze(-1)

            gaussian = 0.7*gaussian_gt + 0.3*gaussian_pt 

            inside_flag = gaussian >= torch.exp(torch.tensor([-self.gauss_thr])).to(device1)
            length = range(assign_result.gt_inds.size(0))
            inside_mask = inside_flag[length, (assign_result.gt_inds-1).clamp(min=0)]
            assign_result.gt_inds *= inside_mask


        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result

    def lvl_assign_wrt_ranking(self,  gt_bboxes, overlaps, gt_labels=None):
        # assign samples inside a certain FPN layer
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
        # assign negative samples

        scale_range = torch.tensor([0,64,128,256,512], device=overlaps.device)
        scale_range = scale_range #** 2
        num_priors = torch.tensor([0,0,0,0,0], device=overlaps.device)
        start_lvl = torch.tensor([0,0,0,0,0], device=overlaps.device)
        end_lvl = torch.tensor([0,0,0,0,0], device=overlaps.device)

        feat_size = 1024/8

        # assign labels according to FPN level

        #overlaps_fpn = overlaps
        overlaps_fpn = torch.zeros_like(overlaps)

        #gt_hbb = self.obb2hbb_le135_xywh(gt_bboxes)

        # assign lvls accodting to its 
        #gt_hbb_scales = gt_hbb[:,2]*gt_hbb[:,3] # assign by hbb scales
        #gt_hbb_scales = gt_bboxes[:,2]*gt_bboxes[:,3]  assign by obb scales
        gt_hbb_scales, _ = torch.max(gt_bboxes[:,2:4], 1)


        gt_hbb_scales[(gt_hbb_scales>scale_range[0]) & (gt_hbb_scales<=scale_range[1])] = 0
        gt_hbb_scales[(gt_hbb_scales>scale_range[1]) & (gt_hbb_scales<=scale_range[2])] = 1
        gt_hbb_scales[(gt_hbb_scales>scale_range[2]) & (gt_hbb_scales<=scale_range[3])] = 2
        gt_hbb_scales[(gt_hbb_scales>scale_range[3]) & (gt_hbb_scales<=scale_range[4])] = 3
        gt_hbb_scales[(gt_hbb_scales>scale_range[4])] = 4

        for lvl in range(5):
            start_lvl[lvl] = end_lvl[lvl-1] + num_priors[lvl] 
            
            num_priors[lvl] = feat_size ** 2
            feat_size = feat_size/2

            end_lvl[lvl] = end_lvl[lvl-1] + num_priors[lvl] 

            #overlaps_lvl = overlaps[:,start_lvl:end_lvl]
        start_lvl = start_lvl.long()
        end_lvl = end_lvl.long()

        # set other levels to zero, set the assigned level to a positive number 
        for i in range(num_gts):
            gt_hbb_lvl = gt_hbb_scales[i].long()
            overlaps_fpn[i,start_lvl[gt_hbb_lvl]:end_lvl[gt_hbb_lvl]] = overlaps[i,start_lvl[gt_hbb_lvl]:end_lvl[gt_hbb_lvl]]

        max_overlaps, argmax_overlaps = overlaps_fpn.max(dim=0)
        # for each gt, topk anchors
        # for each gt, the topk of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps_fpn.topk(self.topk, dim=1, largest=True, sorted=True)  # gt_argmax_overlaps [num_gt, k]

        #print(max_overlaps)

        assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < 1.0)] = 0

        #assign wrt ranking
        for i in range(num_gts):
            for j in range(self.topk):
                max_overlap_inds = overlaps_fpn[i,:] == gt_max_overlaps[i,j]
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
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each gt, topk anchors
        # for each gt, the topk of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.topk(self.topk, dim=1, largest=True, sorted=True)  # gt_argmax_overlaps [num_gt, k]


        assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < 0.8)] = 0

        #assign wrt ranking
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

    def obb2hbb_le135_xywh(self, rotatex_boxes):
        """Convert oriented bounding boxes to horizontal bounding boxes.

        Args:
            obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

        Returns:
            hbbs (torch.Tensor): [x_ctr,y_ctr,w,h,-pi/2]
        """
        polys = self.obb2poly_le135(rotatex_boxes)
        xmin, _ = polys[:, ::2].min(1)
        ymin, _ = polys[:, 1::2].min(1)
        xmax, _ = polys[:, ::2].max(1)
        ymax, _ = polys[:, 1::2].max(1)
        bboxes = torch.stack([xmin, ymin, xmax, ymax], dim=1)
        x_ctr = (bboxes[:, 2] + bboxes[:, 0]) / 2.0
        y_ctr = (bboxes[:, 3] + bboxes[:, 1]) / 2.0
        edges1 = torch.abs(bboxes[:, 2] - bboxes[:, 0])
        edges2 = torch.abs(bboxes[:, 3] - bboxes[:, 1])
        #angles = bboxes.new_zeros(bboxes.size(0))
        inds = edges1 < edges2
        rotated_boxes = torch.stack((x_ctr, y_ctr, edges1, edges2), dim=1)
        rotated_boxes[inds, 2] = edges2[inds]
        rotated_boxes[inds, 3] = edges1[inds]
        #rotated_boxes[inds, 4] = np.pi / 2.0
        return rotated_boxes

    def obb2poly_le135(self, rboxes):
        """Convert oriented bounding boxes to polygons.

        Args:
            obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

        Returns:
            polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
        """
        N = rboxes.shape[0]
        if N == 0:
            return rboxes.new_zeros((rboxes.size(0), 8))
        x_ctr, y_ctr, width, height, angle = rboxes.select(1, 0), rboxes.select(
            1, 1), rboxes.select(1, 2), rboxes.select(1, 3), rboxes.select(1, 4)
        tl_x, tl_y, br_x, br_y = \
            -width * 0.5, -height * 0.5, \
            width * 0.5, height * 0.5
        rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y],
                            dim=0).reshape(2, 4, N).permute(2, 0, 1)
        sin, cos = torch.sin(angle), torch.cos(angle)
        M = torch.stack([cos, -sin, sin, cos], dim=0).reshape(2, 2,
                                                            N).permute(2, 0, 1)
        polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
        polys[:, ::2] += x_ctr.unsqueeze(1)
        polys[:, 1::2] += y_ctr.unsqueeze(1)

        return polys.contiguous()


    