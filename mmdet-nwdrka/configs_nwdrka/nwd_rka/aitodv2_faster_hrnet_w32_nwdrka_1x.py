'''
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.226
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.553
Average Precision  (AP) @[ IoU=0.55      | area=   all | maxDets=1500 ] = 0.490
Average Precision  (AP) @[ IoU=0.60      | area=   all | maxDets=1500 ] = 0.417
Average Precision  (AP) @[ IoU=0.65      | area=   all | maxDets=1500 ] = 0.329
Average Precision  (AP) @[ IoU=0.70      | area=   all | maxDets=1500 ] = 0.235
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.142
Average Precision  (AP) @[ IoU=0.80      | area=   all | maxDets=1500 ] = 0.067
Average Precision  (AP) @[ IoU=0.85      | area=   all | maxDets=1500 ] = 0.022
Average Precision  (AP) @[ IoU=0.90      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.95      | area=   all | maxDets=1500 ] = 0.002
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.084
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.218
Average Precision  (AP) @[ IoU=0.50      | area=  tiny | maxDets=1500 ] = 0.553
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.276
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.360
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.341
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.365
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.369
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.149
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.374
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.415
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.472
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.804
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.287
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.347
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.448
# Class-specific LRP-Optimal Thresholds #
 [0.708 0.757 0.76  0.837 0.914 0.678 0.706 0.81 ]

+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.284 | bridge        | 0.184 | storage-tank | 0.368 |
| ship     | 0.350 | swimming-pool | 0.155 | vehicle      | 0.286 |
| person   | 0.115 | wind-mill     | 0.065 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+

+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.757 | bridge        | 0.844 | storage-tank | 0.682 |
| ship     | 0.693 | swimming-pool | 0.863 | vehicle      | 0.742 |
| person   | 0.903 | wind-mill     | 0.947 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+
OrderedDict([('bbox_mAP', 0.226), ('bbox_mAP_50', 0.553), ('bbox_mAP_75', 0.49), ('bbox_mAP_vt', 0.417), ('bbox_mAP_t', 0.329), ('bbox_mAP_s', 0.235), ('bbox_mAP_m', 0.142),
 ('bbox_oLRP', 0.276), ('bbox_oLRP_Localisation', 0.36), ('bbox_oLRP_false_positive', 0.341),
 ('bbox_oLRP_false_negative', 0.365), ('bbox_mAP_copypaste', '0.226 -1.000 0.553 0.490 0.417 0.329')])
'''
_base_ = ['aitodv2_faster_r50_nwdrka_1x.py']
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w32',
    backbone=dict(
        _delete_=True,
        type='HRNet',
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256)))),
    neck=dict(
        _delete_=True,
        type='HRFPN',
        in_channels=[32, 64, 128, 256],
        out_channels=256))