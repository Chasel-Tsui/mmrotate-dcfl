'''
dilated_rate = 4 - by density, top1
+--------------------+-------+--------+--------+-------+
| class              | gts   | dets   | recall | ap    |
+--------------------+-------+--------+--------+-------+
| plane              | 4859  | 25386  | 0.968  | 0.897 |
| baseball-diamond   | 436   | 6075   | 0.961  | 0.851 |
| bridge             | 893   | 27599  | 0.816  | 0.605 |
| ground-track-field | 265   | 7095   | 0.958  | 0.857 |
| small-vehicle      | 82820 | 507414 | 0.633  | 0.517 |
| large-vehicle      | 10620 | 219201 | 0.905  | 0.759 |
| ship               | 26344 | 155580 | 0.888  | 0.799 |
| tennis-court       | 1539  | 10745  | 0.971  | 0.907 |
| basketball-court   | 292   | 7686   | 0.979  | 0.852 |
| storage-tank       | 5117  | 51533  | 0.803  | 0.708 |
| soccer-ball-field  | 247   | 5404   | 0.806  | 0.653 |
| roundabout         | 338   | 9208   | 0.858  | 0.742 |
| harbor             | 4689  | 105615 | 0.914  | 0.767 |
| swimming-pool      | 1375  | 27221  | 0.783  | 0.640 |
| helicopter         | 128   | 6451   | 0.930  | 0.736 |
| container-crane    | 28    | 14309  | 0.500  | 0.037 |
| airport            | 102   | 5528   | 0.951  | 0.894 |
| helipad            | 4     | 6555   | 1.000  | 0.008 |
+--------------------+-------+--------+--------+-------+
| mAP                |       |        |        | 0.679 |
+--------------------+-------+--------+--------+-------+
2022-09-30 13:22:30,022 - mmrotate - INFO - Exp name: t001.01.06.04_dotav2_retinanet_r101_1x_da4.2_dh_adr2.py
2022-09-30 13:22:30,023 - mmrotate - INFO - Epoch(val) [12][4053]	mAP: 0.6795
'''
_base_ = [
    '../_base_/datasets/dotav2_test.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

angle_version = 'le135'
model = dict(
    type='RotatedRetinaNet',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RDenseHead',
        num_classes=18,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        dilation_rate = 4,
        assign_by_circumhbbox=None,
        anchor_generator=dict(
            type='RotatedAnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=1,
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
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        loss_density = dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='DenseAssigner',
            neg_iou_thr=0.8,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            gpu_assign_thr= 512,
            iou_calculator=dict(type='RBboxMetrics2D'),
            assign_metric='gjsd_10',
            topk= 16,
            dense_thr = 0.2,
            inside_circle= 'rect',
            dense_gauss_thr = None),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.4),
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
checkpoint_config = dict(interval=4)
evaluation = dict(interval=4, metric='mAP')