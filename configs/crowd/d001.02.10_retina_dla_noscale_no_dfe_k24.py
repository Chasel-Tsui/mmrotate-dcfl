'''
+--------------------+-------+--------+--------+-------+
| class              | gts   | dets   | recall | ap    |
+--------------------+-------+--------+--------+-------+
| plane              | 4859  | 33773  | 0.923  | 0.858 |
| baseball-diamond   | 436   | 8394   | 0.911  | 0.769 |
| bridge             | 893   | 53761  | 0.674  | 0.412 |
| ground-track-field | 265   | 10074  | 0.906  | 0.677 |
| small-vehicle      | 82820 | 539840 | 0.578  | 0.418 |
| large-vehicle      | 10620 | 237593 | 0.882  | 0.692 |
| ship               | 26344 | 178010 | 0.827  | 0.743 |
| tennis-court       | 1539  | 17586  | 0.953  | 0.905 |
| basketball-court   | 292   | 10421  | 0.877  | 0.602 |
| storage-tank       | 5117  | 53490  | 0.735  | 0.615 |
| soccer-ball-field  | 247   | 9634   | 0.761  | 0.547 |
| roundabout         | 338   | 10229  | 0.817  | 0.664 |
| harbor             | 4689  | 133035 | 0.855  | 0.665 |
| swimming-pool      | 1375  | 26373  | 0.677  | 0.532 |
| helicopter         | 128   | 6983   | 0.789  | 0.530 |
| container-crane    | 28    | 17354  | 0.250  | 0.002 |
| airport            | 102   | 10952  | 0.902  | 0.695 |
| helipad            | 4     | 8178   | 0.500  | 0.017 |
+--------------------+-------+--------+--------+-------+
| mAP                |       |        |        | 0.575 |
+--------------------+-------+--------+--------+-------+
2022-12-03 23:06:20,459 - mmrotate - INFO - Exp name: d001.02.10_retina_dla_noscale_no_dfe_k24.py
2022-12-03 23:06:20,460 - mmrotate - INFO - Epoch(val) [12][4053]	mAP: 0.5746
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