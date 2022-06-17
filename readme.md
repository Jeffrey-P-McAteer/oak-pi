
# Jeff LAN notes:


```bash
badssh alarm@192.168.5.210

cd /opt/oak-pi

# yay -S python python-pip
# yay -S opencv python-opencv hdf5
python -m server

# yay -S rxvt-unicode xpra xorg
xpra start --bind-tcp=0.0.0.0:14500 --html=on --daemon=no --start=urxvt


```

Backup stored on `/mnt/scratch/oak_pi_dev_mmcblk0`, contains boot wifi & `yay`


# Misc. Research

https://learnopencv.com/depth-perception-using-stereo-camera-python-c/

https://learnopencv.com/object-detection-with-depth-measurement-with-oak-d/

https://docs.openvino.ai/latest/omz_models_group_intel.html

https://docs.openvino.ai/latest/omz_models_model_head_pose_estimation_adas_0001.html#doxid-omz-models-model-head-pose-estimation-adas-0001

https://docs.openvino.ai/latest/omz_models_model_human_pose_estimation_0007.html

https://docs.openvino.ai/latest/omz_demos_human_pose_estimation_demo_cpp.html#doxid-omz-demos-human-pose-estimation-demo-cpp

https://docs.openvino.ai/latest/omz_models_model_faster_rcnn_resnet101_coco_sparse_60_0001.html

https://stackoverflow.com/questions/5105482/compile-main-python-program-using-cython

https://github.com/pyston/pyston

https://docs.openvino.ai/latest/omz_models_model_action_recognition_0001.html

Graphics:
https://threejs.org/examples/#webgl_loader_collada_skinning

https://viso.ai/deep-learning/openpose/



