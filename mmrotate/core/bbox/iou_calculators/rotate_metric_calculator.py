# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops import box_iou_rotated

from .builder import ROTATED_IOU_CALCULATORS
from mmrotate.core.bbox.transforms import hbb2obb



@ROTATED_IOU_CALCULATORS.register_module()
class RBboxMetrics2D(object):
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __call__(self,
                 bboxes1,
                 bboxes2,
                 mode='iou',
                 is_aligned=False,
                 version='oc'):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (torch.Tensor): bboxes have shape (m, 5) in
                <cx, cy, w, h, a> format, or shape (m, 6) in
                 <cx, cy, w, h, a2, score> format.
            bboxes2 (torch.Tensor): bboxes have shape (m, 5) in
                <cx, cy, w, h, a> format, shape (m, 6) in
                 <cx, cy, w, h, a, score> format, or be empty.
                 If ``is_aligned `` is ``True``, then m and n must be equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.
            version (str, optional): Angle representations. Defaults to 'oc'.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 2, 4, 5, 6]
        assert bboxes2.size(-1) in [0, 2, 4, 5, 6]
        
        if bboxes1.size(-1) == 4:
            bboxes1 = hbb2obb(bboxes1, version)
        if bboxes2.size(-1) == 4:
            bboxes2 = hbb2obb(bboxes2, version)

        if bboxes2.size(-1) == 6:
            bboxes2 = bboxes2[..., :5]
        if bboxes1.size(-1) == 6:
            bboxes1 = bboxes1[..., :5]
        return rbbox_metrics(bboxes1.contiguous(), bboxes2.contiguous(), mode,
                              is_aligned)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str


def rbbox_metrics(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    Args:
        bboxes1 (torch.Tensor): shape (B, m, 5) in <cx, cy, w, h, a> format
            or empty.
        bboxes2 (torch.Tensor): shape (B, n, 5) in <cx, cy, w, h, a> format
            or empty.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
    """
    assert mode in ['iou', 'iof','gjsd','center_distance2']
    # Either the boxes are empty or the length of boxes's last dimension is 5
    if mode in ['center_distance2']:
        pass
    else:
        assert (bboxes1.size(-1) == 5 or bboxes1.size(0) == 0)
        assert (bboxes2.size(-1) == 5 or bboxes2.size(0) == 0)

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if mode in ['iou','iof']:
        # resolve `rbbox_overlaps` abnormal when input rbbox is too small.
        clamped_bboxes1 = bboxes1.detach().clone()
        clamped_bboxes2 = bboxes2.detach().clone()
        clamped_bboxes1[:, 2:4].clamp_(min=1e-3)
        clamped_bboxes2[:, 2:4].clamp_(min=1e-3)

        return box_iou_rotated(clamped_bboxes1, clamped_bboxes2, mode, is_aligned)

    if mode == 'gjsd':
        g_bboxes1 = xy_wh_r_2_xy_sigma(bboxes1)
        g_bboxes2 = xy_wh_r_2_xy_sigma(bboxes2)
        gjsd = get_gjsd(g_bboxes1,g_bboxes2)
        distance = 1/(1+gjsd)

        return distance

    if mode == 'center_distance2':
        center1 = bboxes1[..., :, None, :2] 
        center2 = bboxes2[..., None, :, :2] 
        whs = center1[..., :2] - center2[..., :2]

        center_distance2 = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + 1e-6 #

        #distance = torch.sqrt(center_distance2)
    
        return center_distance2


def xy_wh_r_2_xy_sigma(xywhr):
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


def get_gjsd(pred, target, alpha=0.5):
    xy_p, Sigma_p = pred  # mu_1, sigma_1
    xy_t, Sigma_t = target # mu_2, sigma_2

    Sigma_p = Sigma_p.reshape(-1, 2, 2)
    Sigma_t = Sigma_t.reshape(-1, 2, 2)
    
    
    xy_p = xy_p[...,:,None,:2]
    xy_t = xy_t[...,None,:,:2]

    # get the inverse of Sigma_p and Sigma_t
    Sigma_p_inv = torch.stack((Sigma_p[..., 1, 1], -Sigma_p[..., 0, 1],
                               -Sigma_p[..., 1, 0], Sigma_p[..., 0, 0]),
                              dim=-1).reshape(-1, 2, 2)
    Sigma_p_inv = Sigma_p_inv / Sigma_p.det().unsqueeze(-1).unsqueeze(-1)
    Sigma_t_inv = torch.stack((Sigma_t[..., 1, 1], -Sigma_t[..., 0, 1],
                               -Sigma_t[..., 1, 0], Sigma_t[..., 0, 0]),
                              dim=-1).reshape(-1, 2, 2)
    Sigma_t_inv = Sigma_t_inv / Sigma_t.det().unsqueeze(-1).unsqueeze(-1)

    Sigma_p = Sigma_p[...,:,None,:2,:2]
    Sigma_p_inv = Sigma_p_inv[...,:,None,:2,:2]
    Sigma_t = Sigma_t[...,None,:,:2,:2]
    Sigma_t_inv = Sigma_t_inv[...,None,:,:2,:2]
    
    Sigma_alpha_ori = ((1-alpha)*Sigma_p_inv + alpha*Sigma_t_inv)

    # get the inverse of Sigma_alpha_ori, namely Sigma_alpha
    Sigma_alpha =  torch.stack((Sigma_alpha_ori[..., 1, 1], -Sigma_alpha_ori[..., 0, 1],
                               -Sigma_alpha_ori[..., 1, 0], Sigma_alpha_ori[..., 0, 0]),
                              dim=-1).reshape(Sigma_alpha_ori.size(0), Sigma_alpha_ori.size(1), 2, 2)
    Sigma_alpha = Sigma_alpha / Sigma_alpha_ori.det().unsqueeze(-1).unsqueeze(-1)
    # get the inverse of Sigma_alpha, namely Sigma_alpha_inv
    Sigma_alpha_inv = torch.stack((Sigma_alpha[..., 1, 1], -Sigma_alpha[..., 0, 1],
                               -Sigma_alpha[..., 1, 0], Sigma_alpha[..., 0, 0]),
                              dim=-1).reshape(Sigma_alpha.size(0),Sigma_alpha.size(1), 2, 2)
    Sigma_alpha_inv = Sigma_alpha_inv / Sigma_alpha.det().unsqueeze(-1).unsqueeze(-1)

    # mu_alpha
    xy_p = xy_p.unsqueeze(-1)
    xy_t = xy_t.unsqueeze(-1)
    
    mu_alpha_1 = (1-alpha)* Sigma_p_inv.matmul(xy_p) + alpha * Sigma_t_inv.matmul(xy_t)
    mu_alpha = Sigma_alpha.matmul(mu_alpha_1)
    
    # the first part of GJSD 
    first_part = (1-alpha) * xy_p.permute(0,1,3,2).matmul(Sigma_p_inv).matmul(xy_p) + alpha * xy_t.permute(0,1,3,2).matmul(Sigma_t_inv).matmul(xy_t) - mu_alpha.permute(0,1,3,2).matmul(Sigma_alpha_inv).matmul(mu_alpha)
    second_part = ((Sigma_p.det() ** (1-alpha))*(Sigma_t.det() ** alpha))/(Sigma_alpha.det())
    second_part = second_part.log()

    if first_part.is_cuda:
        gjsd = 0.5 * (first_part.half().squeeze(-1).squeeze(-1) + second_part.half())
        #distance = 1/(1+gjsd)
    else:
        gjsd = 0.5 * (first_part.squeeze(-1).squeeze(-1) + second_part)
        #distance = 1/(1+gjsd)

    return gjsd



