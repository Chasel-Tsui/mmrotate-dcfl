'''
+--------------------+-------+--------+--------+-------+
| class              | gts   | dets   | recall | ap    |
+--------------------+-------+--------+--------+-------+
| plane              | 4859  | 32152  | 0.919  | 0.855 |
| baseball-diamond   | 436   | 8788   | 0.924  | 0.773 |
| bridge             | 893   | 56506  | 0.661  | 0.403 |
| ground-track-field | 265   | 9640   | 0.909  | 0.696 |
| small-vehicle      | 82820 | 551139 | 0.601  | 0.453 |
| large-vehicle      | 10620 | 260667 | 0.881  | 0.708 |
| ship               | 26344 | 165457 | 0.858  | 0.786 |
| tennis-court       | 1539  | 14850  | 0.952  | 0.905 |
| basketball-court   | 292   | 9528   | 0.914  | 0.641 |
| storage-tank       | 5117  | 60588  | 0.751  | 0.639 |
| soccer-ball-field  | 247   | 9630   | 0.717  | 0.498 |
| roundabout         | 338   | 13679  | 0.814  | 0.648 |
| harbor             | 4689  | 131126 | 0.860  | 0.674 |
| swimming-pool      | 1375  | 24011  | 0.654  | 0.512 |
| helicopter         | 128   | 8128   | 0.820  | 0.531 |
| container-crane    | 28    | 16099  | 0.286  | 0.001 |
| airport            | 102   | 11364  | 0.902  | 0.694 |
| helipad            | 4     | 7172   | 0.500  | 0.003 |
+--------------------+-------+--------+--------+-------+
| mAP                |       |        |        | 0.579 |
+--------------------+-------+--------+--------+-------+
2022-12-01 16:59:55,158 - mmrotate - INFO - Exp name: d001.02.07_retina_prior_no_dfe_k24.py
2022-12-01 16:59:55,158 - mmrotate - INFO - Epoch(val) [12][4053]	mAP: 0.5790
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
            topk= 24,
            dense_thr = 0.0,
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

#fp16 = dict(loss_scale='dynamic')

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