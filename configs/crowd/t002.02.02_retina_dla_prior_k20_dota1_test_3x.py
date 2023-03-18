'''
+--------------------+-------+-------+--------+-------+
| class              | gts   | dets  | recall | ap    |
+--------------------+-------+-------+--------+-------+
| plane              | 4449  | 15666 | 0.988  | 0.907 |
| baseball-diamond   | 358   | 2815  | 0.992  | 0.901 |
| bridge             | 785   | 13951 | 0.874  | 0.713 |
| ground-track-field | 212   | 2601  | 0.991  | 0.906 |
| small-vehicle      | 10579 | 87651 | 0.928  | 0.798 |
| large-vehicle      | 8819  | 72976 | 0.963  | 0.880 |
| ship               | 18537 | 59734 | 0.943  | 0.885 |
| tennis-court       | 1512  | 5540  | 0.998  | 0.909 |
| basketball-court   | 266   | 2427  | 1.000  | 0.963 |
| storage-tank       | 4740  | 32061 | 0.866  | 0.791 |
| soccer-ball-field  | 251   | 3206  | 0.813  | 0.785 |
| roundabout         | 275   | 2748  | 0.935  | 0.897 |
| harbor             | 4167  | 44203 | 0.968  | 0.888 |
| swimming-pool      | 732   | 6693  | 0.863  | 0.764 |
| helicopter         | 122   | 3377  | 0.984  | 0.894 |
+--------------------+-------+-------+--------+-------+
| mAP                |       |       |        | 0.859 |
+--------------------+-------+-------+--------+-------+
2022-11-29 11:43:19,341 - mmrotate - INFO - Exp name: t002.02.02_retina_dla_prior_k20_dota1_test_3x.py
2022-11-29 11:43:19,341 - mmrotate - INFO - Epoch(val) [36][3066]	mAP: 0.8588

This is your evaluation result for task 1 (VOC metrics):

mAP: 0.7437872354189581
ap of each class: plane:0.8902602023722059, baseball-diamond:0.8177544355404687, bridge:0.4728346218590489, ground-track-field:0.6981105089407851, small-vehicle:0.7940776862471982, large-vehicle:0.7958215454947146, ship:0.8648732802189336, tennis-court:0.9081520007038774, basketball-court:0.8372165391477666, storage-tank:0.8499284215914134, soccer-ball-field:0.6389313551385959, roundabout:0.6579324812922965, harbor:0.7333323101307195, swimming-pool:0.7119157560865252, helicopter:0.4856673865198192
COCO style result:

AP50: 0.7437872354189581
AP75: 0.426489441487596
mAP: 0.4334135718070744
'''
_base_ = [
    '../_base_/datasets/dotav1_test.py', '../_base_/schedules/schedule_3x.py',
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