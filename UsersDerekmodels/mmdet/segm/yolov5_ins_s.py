_base_ = 'yolov5_ins_s-v61_syncbn_fast_8xb16-300e_coco_instance.py'

deepen_factor = 0.33
widen_factor = 0.50

model = dict(
    data_preprocessor=dict(bgr_to_rgb=False),
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))
