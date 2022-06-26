
# Jeff LAN notes:


```bash
badssh alarm@192.168.5.210

# Config for OAK-D lite hardware
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

cd /opt/oak-pi

# yay -S python python-pip
# yay -S opencv python-opencv hdf5
# yay -S open3d-git python-open3d-git
python -m server

/home/alarm/.local/bin/omz_downloader

# Never actually used, but remote x11 over html5. Kinda.
# yay -S rxvt-unicode xpra xpra-html5-git xorg
xpra start --bind-tcp=0.0.0.0:14500 --html=on --daemon=no --speaker=off --microphone=off --start=urxvt

# Nifty file copy
rsync -avz ./ alarm@192.168.5.210:/opt/oak-pi/

# Improved one-liner
rsync -avz ./ alarm@192.168.5.210:/opt/oak-pi/ ; badssh alarm@192.168.5.210 bash -c 'pkill -9 python ; cd /opt/oak-pi ; python -u -m server'


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



https://docs.luxonis.com/projects/api/en/latest/components/pipeline/


https://github.com/geaxgx/depthai_blazepose





