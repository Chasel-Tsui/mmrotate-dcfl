'''
+--------------------+-------+--------+--------+-------+
| class              | gts   | dets   | recall | ap    |
+--------------------+-------+--------+--------+-------+
| plane              | 4859  | 35583  | 0.917  | 0.855 |
| baseball-diamond   | 436   | 12771  | 0.913  | 0.756 |
| bridge             | 893   | 61558  | 0.682  | 0.394 |
| ground-track-field | 265   | 10082  | 0.928  | 0.686 |
| small-vehicle      | 82820 | 617143 | 0.576  | 0.442 |
| large-vehicle      | 10620 | 284766 | 0.872  | 0.689 |
| ship               | 26344 | 194557 | 0.853  | 0.783 |
| tennis-court       | 1539  | 20516  | 0.943  | 0.894 |
| basketball-court   | 292   | 13824  | 0.877  | 0.576 |
| storage-tank       | 5117  | 71996  | 0.747  | 0.639 |
| soccer-ball-field  | 247   | 12442  | 0.717  | 0.491 |
| roundabout         | 338   | 11608  | 0.796  | 0.615 |
| harbor             | 4689  | 148889 | 0.842  | 0.632 |
| swimming-pool      | 1375  | 28453  | 0.666  | 0.523 |
| helicopter         | 128   | 8430   | 0.812  | 0.483 |
| container-crane    | 28    | 19984  | 0.179  | 0.003 |
| airport            | 102   | 15672  | 0.824  | 0.603 |
| helipad            | 4     | 7767   | 0.500  | 0.004 |
+--------------------+-------+--------+--------+-------+
| mAP                |       |        |        | 0.559 |
+--------------------+-------+--------+--------+-------+
2022-11-26 01:05:29,020 - mmrotate - INFO - Exp name: d001.02.01_retina_dla_prior_k16_c0.4_dotav2.py
2022-11-26 01:05:29,021 - mmrotate - INFO - Epoch(val) [12][4053]	mAP: 0.5594
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
            type='DenseAssigner',
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