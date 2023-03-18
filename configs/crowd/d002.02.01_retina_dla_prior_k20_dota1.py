'''
+--------------------+-------+--------+--------+-------+
| class              | gts   | dets   | recall | ap    |
+--------------------+-------+--------+--------+-------+
| plane              | 4449  | 30686  | 0.950  | 0.889 |
| baseball-diamond   | 358   | 10617  | 0.919  | 0.747 |
| bridge             | 785   | 44567  | 0.682  | 0.399 |
| ground-track-field | 212   | 8842   | 0.892  | 0.606 |
| small-vehicle      | 10579 | 158949 | 0.878  | 0.681 |
| large-vehicle      | 8819  | 133993 | 0.912  | 0.755 |
| ship               | 18537 | 98901  | 0.905  | 0.824 |
| tennis-court       | 1512  | 13015  | 0.950  | 0.907 |
| basketball-court   | 266   | 8223   | 0.910  | 0.655 |
| storage-tank       | 4740  | 36665  | 0.758  | 0.647 |
| soccer-ball-field  | 251   | 9775   | 0.741  | 0.522 |
| roundabout         | 275   | 10967  | 0.785  | 0.631 |
| harbor             | 4167  | 108945 | 0.883  | 0.688 |
| swimming-pool      | 732   | 14098  | 0.727  | 0.519 |
| helicopter         | 122   | 7189   | 0.811  | 0.471 |
+--------------------+-------+--------+--------+-------+
| mAP                |       |        |        | 0.663 |
+--------------------+-------+--------+--------+-------+
2022-11-28 00:31:41,700 - mmrotate - INFO - Exp name: d002.02.01_retina_dla_prior_k20_dota1.py
2022-11-28 00:31:41,700 - mmrotate - INFO - Epoch(val) [12][3066]	mAP: 0.6628
'''
_base_ = [
    '../_base_/datasets/dotav1_val.py', '../_base_/schedules/schedule_1x.py',
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
        num_classes=15,
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
            topk= 20,
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