'''
+-------------------------+-------+--------+--------+-------+
| class                   | gts   | dets   | recall | ap    |
+-------------------------+-------+--------+--------+-------+
| airplane                | 8212  | 17991  | 0.760  | 0.698 |
| airport                 | 666   | 9558   | 0.817  | 0.494 |
| baseballfield           | 3434  | 10053  | 0.845  | 0.773 |
| basketballcourt         | 2146  | 11064  | 0.914  | 0.859 |
| bridge                  | 2587  | 49387  | 0.615  | 0.388 |
| chimney                 | 1031  | 10343  | 0.832  | 0.757 |
| expressway-service-area | 1085  | 8553   | 0.920  | 0.837 |
| expressway-toll-station | 688   | 9235   | 0.815  | 0.704 |
| dam                     | 538   | 14911  | 0.801  | 0.375 |
| golffield               | 575   | 7602   | 0.920  | 0.784 |
| groundtrackfield        | 1885  | 15732  | 0.951  | 0.810 |
| harbor                  | 3102  | 87212  | 0.722  | 0.457 |
| overpass                | 1780  | 27322  | 0.716  | 0.545 |
| ship                    | 35184 | 90966  | 0.926  | 0.881 |
| stadium                 | 672   | 6685   | 0.912  | 0.717 |
| storagetank             | 23361 | 46170  | 0.764  | 0.682 |
| tenniscourt             | 7343  | 14122  | 0.900  | 0.859 |
| trainstation            | 509   | 7863   | 0.835  | 0.576 |
| vehicle                 | 26613 | 221902 | 0.625  | 0.504 |
| windmill                | 2998  | 7862   | 0.799  | 0.647 |
+-------------------------+-------+--------+--------+-------+
| mAP                     |       |        |        | 0.667 |
+-------------------------+-------+--------+--------+-------+
'''
_base_ = [
    '../_base_/datasets/diorr.py', '../_base_/schedules/schedule_40e.py',
    '../_base_/default_runtime.py'
]

angle_version = 'le135'
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    type='RotatedRepPoints',
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
        num_outs=5,
        norm_cfg=norm_cfg),
    bbox_head=dict(
        type='OrientedRepPointsHead',
        num_classes=20,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        num_points=9,
        gradient_mul=0.3,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=2,
        norm_cfg=norm_cfg,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_init=dict(type='ConvexGIoULoss', loss_weight=0.375),
        loss_bbox_refine=dict(type='ConvexGIoULoss', loss_weight=1.0),
        loss_spatial_init=dict(type='SpatialBorderLoss', loss_weight=0.05),
        loss_spatial_refine=dict(type='SpatialBorderLoss', loss_weight=0.1),
        init_qua_weight=0.2,
        top_ratio=0.4),
    # training and testing settings
    train_cfg=dict(
        init=dict(
            assigner=dict(type='ConvexAssigner', scale=4, pos_num=1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        refine=dict(
            assigner=dict(
                type='MaxConvexIoUAssigner',
                pos_iou_thr=0.1,
                neg_iou_thr=0.1,
                min_pos_iou=0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)),
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
    dict(type='RResize', img_scale=(800, 800)),
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
optimizer = dict(lr=0.008)
checkpoint_config = dict(interval=4)
