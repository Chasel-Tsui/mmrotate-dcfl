'''
SGD
+--------------------+-------+--------+--------+-------+
| class              | gts   | dets   | recall | ap    |
+--------------------+-------+--------+--------+-------+
| plane              | 4859  | 19408  | 0.883  | 0.809 |
| baseball-diamond   | 436   | 6691   | 0.897  | 0.741 |
| bridge             | 893   | 24254  | 0.557  | 0.363 |
| ground-track-field | 265   | 11838  | 0.917  | 0.654 |
| small-vehicle      | 82820 | 295901 | 0.498  | 0.344 |
| large-vehicle      | 10620 | 200506 | 0.770  | 0.542 |
| ship               | 26344 | 129739 | 0.813  | 0.747 |
| tennis-court       | 1539  | 14236  | 0.939  | 0.895 |
| basketball-court   | 292   | 7326   | 0.736  | 0.540 |
| storage-tank       | 5117  | 37018  | 0.679  | 0.586 |
| soccer-ball-field  | 247   | 6633   | 0.729  | 0.489 |
| roundabout         | 338   | 14714  | 0.808  | 0.614 |
| harbor             | 4689  | 38174  | 0.653  | 0.489 |
| swimming-pool      | 1375  | 18887  | 0.626  | 0.511 |
| helicopter         | 128   | 11084  | 0.758  | 0.360 |
| container-crane    | 28    | 579    | 0.000  | 0.000 |
| airport            | 102   | 6961   | 0.637  | 0.517 |
| helipad            | 4     | 8096   | 0.000  | 0.000 |
+--------------------+-------+--------+--------+-------+
| mAP                |       |        |        | 0.511 |
+--------------------+-------+--------+--------+-------+
2022-09-22 09:11:49,484 - mmrotate - INFO - Exp name: dotav2_rotated_retinanet_obb_convnext_sgd_fpn_1x_dota_le135.py
2022-09-22 09:11:49,484 - mmrotate - INFO - Epoch(val) [12][4053]	mAP: 0.5112
'''
_base_ = [
    '../_base_/datasets/dotav2_val.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# please install mmcls>=0.22.0
# import mmcls.models to trigger register_module in mmcls
custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth'  # noqa


angle_version = 'le135'
model = dict(
    type='RotatedRetinaNet',
    backbone=dict(
        type='mmcls.ConvNeXt',
        arch='tiny',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RotatedRetinaHead',
        num_classes=18,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        assign_by_circumhbbox=None,
        anchor_generator=dict(
            type='RotatedAnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[1.0, 0.5, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range=angle_version,
            norm_factor=None,
            edge_swap=False,
            proj_xy=False,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=dict(
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


find_unused_parameters = True

evaluation = dict(interval=4, metric='mAP')
# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=4)