# from https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov8
# copy of the original source https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov8/yolov8_n_syncbn_fast_8xb16-500e_coco.py
_base_ = './yolov8_s_syncbn_fast_8xb16-500e_coco.py'

deepen_factor = 0.33
widen_factor = 0.25

model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))
