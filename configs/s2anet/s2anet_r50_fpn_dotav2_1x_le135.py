'''
+--------------------+-------+--------+--------+-------+
| class              | gts   | dets   | recall | ap    |
+--------------------+-------+--------+--------+-------+
| plane              | 4859  | 12080  | 0.906  | 0.866 |
| baseball-diamond   | 436   | 5851   | 0.892  | 0.715 |
| bridge             | 893   | 22446  | 0.581  | 0.406 |
| ground-track-field | 265   | 6346   | 0.917  | 0.732 |
| small-vehicle      | 82820 | 272177 | 0.566  | 0.467 |
| large-vehicle      | 10620 | 141645 | 0.863  | 0.732 |
| ship               | 26344 | 98707  | 0.867  | 0.807 |
| tennis-court       | 1539  | 10812  | 0.945  | 0.905 |
| basketball-court   | 292   | 4971   | 0.884  | 0.636 |
| storage-tank       | 5117  | 28158  | 0.758  | 0.682 |
| soccer-ball-field  | 247   | 5998   | 0.709  | 0.502 |
| roundabout         | 338   | 11962  | 0.796  | 0.639 |
| harbor             | 4689  | 40578  | 0.732  | 0.627 |
| swimming-pool      | 1375  | 14325  | 0.664  | 0.545 |
| helicopter         | 128   | 12990  | 0.727  | 0.460 |
| container-crane    | 28    | 16119  | 0.071  | 0.000 |
| airport            | 102   | 6576   | 0.725  | 0.621 |
| helipad            | 4     | 7051   | 0.500  | 0.164 |
+--------------------+-------+--------+--------+-------+
| mAP                |       |        |        | 0.584 |
+--------------------+-------+--------+--------+-------+
2022-05-05 00:39:48,459 - mmrotate - INFO - Exp name: s2anet_r50_fpn_dotav2_1x_le135.py
2022-05-05 00:39:48,459 - mmrotate - INFO - Epoch(val) [12][4053]	mAP: 0.5836
'''
_base_ = [
    '../_base_/datasets/dotav2_val.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

angle_version = 'le135'
model = dict(
    type='S2ANet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    fam_head=dict(
        type='RotatedRetinaHead',
        num_classes=18,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        assign_by_circumhbbox=None,
        anchor_generator=dict(
            type='RotatedAnchorGenerator',
            scales=[4],
            ratios=[1.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range=angle_version,
            norm_factor=1,
            edge_swap=False,
            proj_xy=True,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)),
    align_cfgs=dict(
        type='AlignConv',
        kernel_size=3,
        channels=256,
        featmap_strides=[8, 16, 32, 64, 128]),
    odm_head=dict(
        type='ODMRefineHead',
        num_classes=18,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        assign_by_circumhbbox=None,
        anchor_generator=dict(
            type='PseudoAnchorGenerator', strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range=angle_version,
            norm_factor=1,
            edge_swap=False,
            proj_xy=True,
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)),
    train_cfg=dict(
        fam_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBboxOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        odm_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBboxOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
data = dict(
    train=dict(pipeline=train_pipeline, version=angle_version),
    val=dict(version=angle_version),
    test=dict(version=angle_version))
checkpoint_config = dict(interval=2)
evaluation = dict(interval=4, metric='mAP')
