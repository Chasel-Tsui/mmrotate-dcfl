'''
+--------------------+-------+--------+--------+-------+
| class              | gts   | dets   | recall | ap    |
+--------------------+-------+--------+--------+-------+
| plane              | 4859  | 35501  | 0.911  | 0.840 |
| baseball-diamond   | 436   | 22723  | 0.881  | 0.624 |
| bridge             | 893   | 52914  | 0.629  | 0.363 |
| ground-track-field | 265   | 13024  | 0.913  | 0.634 |
| small-vehicle      | 82820 | 598039 | 0.585  | 0.444 |
| large-vehicle      | 10620 | 267928 | 0.879  | 0.700 |
| ship               | 26344 | 191312 | 0.856  | 0.782 |
| tennis-court       | 1539  | 15003  | 0.953  | 0.903 |
| basketball-court   | 292   | 14036  | 0.873  | 0.598 |
| storage-tank       | 5117  | 70161  | 0.748  | 0.635 |
| soccer-ball-field  | 247   | 15088  | 0.737  | 0.474 |
| roundabout         | 338   | 14623  | 0.805  | 0.618 |
| harbor             | 4689  | 141166 | 0.849  | 0.642 |
| swimming-pool      | 1375  | 27560  | 0.676  | 0.521 |
| helicopter         | 128   | 10771  | 0.828  | 0.505 |
| container-crane    | 28    | 15233  | 0.107  | 0.000 |
| airport            | 102   | 14148  | 0.824  | 0.530 |
| helipad            | 4     | 5944   | 0.500  | 0.001 |
+--------------------+-------+--------+--------+-------+
| mAP                |       |        |        | 0.545 |
+--------------------+-------+--------+--------+-------+
2022-11-25 05:26:52,134 - mmrotate - INFO - Exp name: d001.01.03_retina_dla_k16_c0.4_dotav2.py
2022-11-25 05:26:52,134 - mmrotate - INFO - Epoch(val) [12][4053]	mAP: 0.5453
'''

_base_ = [
    '../_base_/datasets/dotav2_val.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

angle_version = 'le135'
model = dict(
    type='RotatedRetinaNet',
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
    bbox_head=dict(
        type='RDenseHead',
        num_classes=18,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
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
        use_qwl=False,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='DensityAssigner',
            ignore_iof_thr=-1,
            gpu_assign_thr= 512,
            iou_calculator=dict(type='RBboxMetrics2D'),
            assign_metric='gjsd_10',
            topk= 16,
            dense_thr = 0.4,
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
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline, version=angle_version),
    val=dict(version=angle_version),
    test=dict(version=angle_version))
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
checkpoint_config = dict(interval=4)
evaluation = dict(interval=4, metric='mAP')