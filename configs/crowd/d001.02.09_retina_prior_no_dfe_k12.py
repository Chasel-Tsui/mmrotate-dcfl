'''
+--------------------+-------+--------+--------+-------+
| class              | gts   | dets   | recall | ap    |
+--------------------+-------+--------+--------+-------+
| plane              | 4859  | 25584  | 0.916  | 0.862 |
| baseball-diamond   | 436   | 8076   | 0.920  | 0.734 |
| bridge             | 893   | 45168  | 0.654  | 0.398 |
| ground-track-field | 265   | 8797   | 0.928  | 0.694 |
| small-vehicle      | 82820 | 549833 | 0.608  | 0.469 |
| large-vehicle      | 10620 | 225310 | 0.886  | 0.709 |
| ship               | 26344 | 161602 | 0.871  | 0.790 |
| tennis-court       | 1539  | 14939  | 0.947  | 0.902 |
| basketball-court   | 292   | 9456   | 0.897  | 0.654 |
| storage-tank       | 5117  | 56954  | 0.766  | 0.656 |
| soccer-ball-field  | 247   | 9201   | 0.749  | 0.499 |
| roundabout         | 338   | 11665  | 0.825  | 0.664 |
| harbor             | 4689  | 111450 | 0.854  | 0.639 |
| swimming-pool      | 1375  | 23275  | 0.674  | 0.537 |
| helicopter         | 128   | 9262   | 0.812  | 0.499 |
| container-crane    | 28    | 23180  | 0.071  | 0.000 |
| airport            | 102   | 8941   | 0.912  | 0.633 |
| helipad            | 4     | 7196   | 0.500  | 0.002 |
+--------------------+-------+--------+--------+-------+
| mAP                |       |        |        | 0.574 |
+--------------------+-------+--------+--------+-------+
2022-12-02 22:57:36,256 - mmrotate - INFO - Exp name: d001.02.09_retina_prior_no_dfe_k12.py
2022-12-02 22:57:36,256 - mmrotate - INFO - Epoch(val) [12][4053]	mAP: 0.5745
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
            topk= 12,
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