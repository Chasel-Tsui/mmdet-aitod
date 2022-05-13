'''
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.285 | bridge        | 0.167 | storage-tank | 0.352 |
| ship     | 0.338 | swimming-pool | 0.127 | vehicle      | 0.263 |
| person   | 0.104 | wind-mill     | 0.073 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+
2021-11-07 15:33:40,321 - mmdet - INFO -
+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.760 | bridge        | 0.857 | storage-tank | 0.697 |
| ship     | 0.708 | swimming-pool | 0.876 | vehicle      | 0.773 |
| person   | 0.913 | wind-mill     | 0.945 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+
2021-11-07 15:33:41,065 - mmdet - INFO - Exp name: v001.04.13_aitodv2_faster_r50_k2_rpn_NWD_rk.py
2021-11-07 15:33:41,065 - mmdet - INFO - Epoch(val) [12][7004]	bbox_mAP: 0.2140, bbox_mAP_50: 0.5320, bbox_mAP_75: 0.4740, bbox_mAP_vt: 0.4000, bbox_mAP_t: 0.3070, bbox_mAP_s: 0.2130, bbox_mAP_m: 0.1250, bbox_oLRP: 0.2680, bbox_oLRP_Localisation: 0.3520, bbox_oLRP_false_positive: 0.3300, bbox_oLRP_false_negative: 0.3530, bbox_mAP_copypaste: 0.214 -1.000 0.532 0.474 0.400 0.307
Loading and preparing results...
DONE (t=6.80s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=2168.28s).
Accumulating evaluation results...
DONE (t=30.57s).
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.214
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.532
Average Precision  (AP) @[ IoU=0.55      | area=   all | maxDets=1500 ] = 0.474
Average Precision  (AP) @[ IoU=0.60      | area=   all | maxDets=1500 ] = 0.400
Average Precision  (AP) @[ IoU=0.65      | area=   all | maxDets=1500 ] = 0.307
Average Precision  (AP) @[ IoU=0.70      | area=   all | maxDets=1500 ] = 0.213
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.125
Average Precision  (AP) @[ IoU=0.80      | area=   all | maxDets=1500 ] = 0.061
Average Precision  (AP) @[ IoU=0.85      | area=   all | maxDets=1500 ] = 0.019
Average Precision  (AP) @[ IoU=0.90      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.95      | area=   all | maxDets=1500 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.077
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.207
Average Precision  (AP) @[ IoU=0.50      | area=  tiny | maxDets=1500 ] = 0.542
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.268
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.352
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.330
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.353
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.357
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.134
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.360
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.414
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.459
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.816
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.291
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.351
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.479
# Class-specific LRP-Optimal Thresholds #
 [0.718 0.817 0.792 0.847 0.905 0.706 0.746 0.833]

 #### NWD evaluation
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.336
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.617
Average Precision  (AP) @[ IoU=0.55      | area=   all | maxDets=1500 ] = 0.597
Average Precision  (AP) @[ IoU=0.60      | area=   all | maxDets=1500 ] = 0.560
Average Precision  (AP) @[ IoU=0.65      | area=   all | maxDets=1500 ] = 0.508
Average Precision  (AP) @[ IoU=0.70      | area=   all | maxDets=1500 ] = 0.428
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.326
Average Precision  (AP) @[ IoU=0.80      | area=   all | maxDets=1500 ] = 0.207
Average Precision  (AP) @[ IoU=0.85      | area=   all | maxDets=1500 ] = 0.093
Average Precision  (AP) @[ IoU=0.90      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.95      | area=   all | maxDets=1500 ] = 0.001
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.300
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.386
Average Precision  (AP) @[ IoU=0.50      | area=  tiny | maxDets=1500 ] = 0.648
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.314
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.199
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.489
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.532
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.542
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.441
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.596
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.469
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.287
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.729
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.223
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.308
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.417
# Class-specific LRP-Optimal Thresholds #
 [0.718 0.809 0.731 0.819 0.905 0.644 0.673 0.859]

+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.218 | bridge        | 0.275 | storage-tank | 0.507 |
| ship     | 0.521 | swimming-pool | 0.151 | vehicle      | 0.487 |
| person   | 0.283 | wind-mill     | 0.244 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+

+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.801 | bridge        | 0.777 | storage-tank | 0.590 |
| ship     | 0.553 | swimming-pool | 0.867 | vehicle      | 0.604 |
| person   | 0.796 | wind-mill     | 0.840 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+
OrderedDict([('bbox_mAP', 0.336), ('bbox_mAP_50', 0.617), ('bbox_mAP_75', 0.597), ('bbox_mAP_vt', 0.56), ('bbox_mAP_t', 0.508), ('bbox_mAP_s', 0.428), ('bbox_mAP_m', 0.326), ('bbox_oLRP', 0.314), ('bbox_oLRP_Localisation', 0.199), ('bbox_oLRP_false_positive', 0.489), ('bbox_oLRP_false_negative', 0.532), ('bbox_mAP_copypaste', '0.336 -1.000 0.617 0.597 0.560 0.508')])


'''
_base_ = [
    #'../_base_/models/faster_rcnn_r50_fpn_aitod.py',
    '../_base_/datasets/aitodv2_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]



model = dict(
    type='FasterRCNN',
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
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=8,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='RankingAssigner',
                ignore_iof_thr=-1,
                gpu_assign_thr=512,
                iou_calculator=dict(type='BboxDistanceMetric'),
                assign_metric='nwd',
                topk=2),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1,
                gpu_assign_thr=512,
                iou_calculator=dict(type='BboxDistanceMetric'),
                assign_metric='iou'),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=3000)
    ))

#fp16 = dict(loss_scale=512.)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[16, 22])
# runner = dict(type='EpochBasedRunner', max_epochs=24)
evaluation = dict(interval=12, metric='bbox')