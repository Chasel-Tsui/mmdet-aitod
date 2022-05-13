'''
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.001 | bridge        | 0.138 | storage-tank | 0.143 |
| ship     | 0.285 | swimming-pool | 0.054 | vehicle      | 0.165 |
| person   | 0.047 | wind-mill     | 0.007 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+
2021-10-30 06:18:27,781 - mmdet - INFO -
+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.996 | bridge        | 0.881 | storage-tank | 0.876 |
| ship     | 0.737 | swimming-pool | 0.943 | vehicle      | 0.849 |
| person   | 0.949 | wind-mill     | 0.983 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+
2021-10-30 06:18:30,617 - mmdet - INFO - Exp name: v001.06.10_aitodv2_retinanet_r50_rpn_NWD_rk.py
2021-10-30 06:18:30,618 - mmdet - INFO - Epoch(val) [12][7004]	bbox_mAP: 0.1050, bbox_mAP_50: 0.2850, bbox_mAP_75: 0.2440, bbox_mAP_vt: 0.1940, bbox_mAP_t: 0.1430, bbox_mAP_s: 0.0950, bbox_mAP_m: 0.0520, bbox_oLRP: 0.1520, bbox_oLRP_Localisation: 0.1420, bbox_oLRP_false_positive: 0.2260, bbox_oLRP_false_negative: 0.2470, bbox_mAP_copypaste: 0.105 -1.000 0.285 0.244 0.194 0.143
Loading and preparing results...
DONE (t=29.82s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=6166.97s).
Accumulating evaluation results...
DONE (t=100.34s).
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.105
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.285
Average Precision  (AP) @[ IoU=0.55      | area=   all | maxDets=1500 ] = 0.244
Average Precision  (AP) @[ IoU=0.60      | area=   all | maxDets=1500 ] = 0.194
Average Precision  (AP) @[ IoU=0.65      | area=   all | maxDets=1500 ] = 0.143
Average Precision  (AP) @[ IoU=0.70      | area=   all | maxDets=1500 ] = 0.095
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.052
Average Precision  (AP) @[ IoU=0.80      | area=   all | maxDets=1500 ] = 0.025
Average Precision  (AP) @[ IoU=0.85      | area=   all | maxDets=1500 ] = 0.010
Average Precision  (AP) @[ IoU=0.90      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.95      | area=   all | maxDets=1500 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.035
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.110
Average Precision  (AP) @[ IoU=0.50      | area=  tiny | maxDets=1500 ] = 0.308
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.152
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.142
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.226
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.247
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.257
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.097
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.280
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.301
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.288
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.902
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.312
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.602
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.655
# Class-specific LRP-Optimal Thresholds #
 [0.105 0.388 0.349 0.435 0.342 0.445 0.398 0.168]
'''
_base_ = [
    #'../_base_/models/retinanet_r50_fpn_aitod.py',
    '../_base_/datasets/aitodv2_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]


# optimizer
model = dict(
    type='RetinaNet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=8,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='RankingAssigner',
            ignore_iof_thr=-1,
            gpu_assign_thr=512,
            iou_calculator=dict(type='BboxDistanceMetric'),
            assign_metric='nwd',
            topk=3),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=3000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=3000))
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)