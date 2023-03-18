'''
dilated_rate = 4 - by density, top1
+--------------------+-------+--------+--------+-------+
| class              | gts   | dets   | recall | ap    |
+--------------------+-------+--------+--------+-------+
| plane              | 4859  | 25406  | 0.964  | 0.897 |
| baseball-diamond   | 436   | 7715   | 0.966  | 0.853 |
| bridge             | 893   | 34219  | 0.786  | 0.552 |
| ground-track-field | 265   | 7257   | 0.962  | 0.857 |
| small-vehicle      | 82820 | 507634 | 0.622  | 0.502 |
| large-vehicle      | 10620 | 208437 | 0.900  | 0.735 |
| ship               | 26344 | 163109 | 0.883  | 0.797 |
| tennis-court       | 1539  | 12760  | 0.961  | 0.908 |
| basketball-court   | 292   | 8025   | 0.983  | 0.852 |
| storage-tank       | 5117  | 64032  | 0.798  | 0.689 |
| soccer-ball-field  | 247   | 6741   | 0.814  | 0.645 |
| roundabout         | 338   | 10259  | 0.861  | 0.721 |
| harbor             | 4689  | 113308 | 0.907  | 0.735 |
| swimming-pool      | 1375  | 28384  | 0.787  | 0.630 |
| helicopter         | 128   | 9921   | 0.938  | 0.712 |
| container-crane    | 28    | 15005  | 0.607  | 0.042 |
| airport            | 102   | 7581   | 0.951  | 0.885 |
| helipad            | 4     | 6299   | 1.000  | 0.025 |
+--------------------+-------+--------+--------+-------+
| mAP                |       |        |        | 0.669 |
+--------------------+-------+--------+--------+-------+
2022-09-30 03:39:02,697 - mmrotate - INFO - Exp name: t001.01.06.04_dotav2_retinanet_r50_1x_da4.2_dh_adr2.py
2022-09-30 03:39:02,697 - mmrotate - INFO - Epoch(val) [12][4053]	mAP: 0.6688

NMS=0.2
+--------------------+-------+--------+--------+-------+
| class              | gts   | dets   | recall | ap    |
+--------------------+-------+--------+--------+-------+
| plane              | 4859  | 16338  | 0.952  | 0.905 |
| baseball-diamond   | 436   | 5505   | 0.954  | 0.860 |
| bridge             | 893   | 23334  | 0.729  | 0.550 |
| ground-track-field | 265   | 5879   | 0.958  | 0.857 |
| small-vehicle      | 82820 | 366383 | 0.589  | 0.480 |
| large-vehicle      | 10620 | 165866 | 0.876  | 0.746 |
| ship               | 26344 | 118064 | 0.870  | 0.801 |
| tennis-court       | 1539  | 9713   | 0.958  | 0.908 |
| basketball-court   | 292   | 5575   | 0.962  | 0.874 |
| storage-tank       | 5117  | 47410  | 0.771  | 0.695 |
| soccer-ball-field  | 247   | 4841   | 0.781  | 0.639 |
| roundabout         | 338   | 9049   | 0.852  | 0.719 |
| harbor             | 4689  | 58132  | 0.842  | 0.731 |
| swimming-pool      | 1375  | 19001  | 0.741  | 0.630 |
| helicopter         | 128   | 8241   | 0.906  | 0.696 |
| container-crane    | 28    | 10603  | 0.429  | 0.063 |
| airport            | 102   | 5003   | 0.922  | 0.896 |
| helipad            | 4     | 5713   | 1.000  | 0.026 |
+--------------------+-------+--------+--------+-------+
| mAP                |       |        |        | 0.671 |
+--------------------+-------+--------+--------+-------+
{'mAP': 0.6707866191864014}

NMS=0.1
+--------------------+-------+--------+--------+-------+
| class              | gts   | dets   | recall | ap    |
+--------------------+-------+--------+--------+-------+
| plane              | 4859  | 14389  | 0.937  | 0.903 |
| baseball-diamond   | 436   | 5030   | 0.952  | 0.861 |
| bridge             | 893   | 20017  | 0.728  | 0.552 |
| ground-track-field | 265   | 5453   | 0.951  | 0.857 |
| small-vehicle      | 82820 | 311777 | 0.575  | 0.481 |
| large-vehicle      | 10620 | 154156 | 0.868  | 0.747 |
| ship               | 26344 | 105379 | 0.863  | 0.802 |
| tennis-court       | 1539  | 9028   | 0.958  | 0.908 |
| basketball-court   | 292   | 4993   | 0.962  | 0.880 |
| storage-tank       | 5117  | 41315  | 0.764  | 0.698 |
| soccer-ball-field  | 247   | 4340   | 0.781  | 0.637 |
| roundabout         | 338   | 8767   | 0.852  | 0.717 |
| harbor             | 4689  | 42958  | 0.824  | 0.730 |
| swimming-pool      | 1375  | 15965  | 0.739  | 0.632 |
| helicopter         | 128   | 7790   | 0.906  | 0.692 |
| container-crane    | 28    | 9165   | 0.214  | 0.059 |
| airport            | 102   | 4331   | 0.902  | 0.897 |
| helipad            | 4     | 5516   | 1.000  | 0.026 |
+--------------------+-------+--------+--------+-------+
| mAP                |       |        |        | 0.671 |
+--------------------+-------+--------+--------+-------+
{'mAP': 0.6709824204444885}
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
        nms=dict(iou_thr=0.2), # 0.4
        max_per_img=1500))

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