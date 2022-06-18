#!/usr/bin/env python

import os
import sys
import subprocess
import traceback
import socket
import platform
import signal
import asyncio
import string
import re

# DepthAI config request (from https://github.com/luxonis/depthai/blob/3819aa513f58f2d749e5d5c94953ce1d2fe0a061/depthai_demo.py )
if platform.machine() == 'aarch64':  # Jetson
  os.environ['OPENBLAS_CORETYPE'] = "ARMV8"

# python -m pip install --user aiohttp
try:
  import aiohttp.web
except:
  traceback.print_exc()
  subprocess.run([
    sys.executable,
    *('-m pip install --user aiohttp'.split(' '))
  ])
  import aiohttp.web

# python -m pip install --user opencv-python # or opencv-contrib-python
try:
  import cv2
except:
  traceback.print_exc()
  subprocess.run([
    sys.executable,
    *('-m pip install --user opencv-contrib-python'.split(' '))
  ])
  import cv2

# python -m pip install --user depthai # interface to OAK-D camera
try:
  import depthai
except:
  traceback.print_exc()
  subprocess.run([
    sys.executable,
    *('-m pip install --user depthai'.split(' '))
  ])
  import depthai

# python -m pip install --user --no-dependencies openvino-dev # intel's model stuff, mostly useful because it provides omz_downloader
try:
  import openvino
except:
  traceback.print_exc()
  subprocess.run([
    sys.executable, # manual "dependency" resolution
    *('-m pip install --user addict defusedxml imagecodecs jstyleson lmdb networkx nibabel nltk openvino openvino-telemetry pandas parasail py-cpuinfo pyclipper pydicom pyyaml rawpy scikit-image scikit-learn sentencepiece shapely texttable  tokenizers tqdm transformers numpy scipy'.split(' '))
  ])
  # Rest of openvino-dev deps:
  # python -m pip install --user addict defusedxml imagecodecs jstyleson lmdb networkx nibabel nltk openvino openvino-telemetry pandas parasail py-cpuinfo pyclipper pydicom pyyaml rawpy scikit-image scikit-learn sentencepiece shapely texttable  tokenizers tqdm transformers numpy scipy

  # TODO add building fast-ctc-decode manually b/c there's no aarch64 distribution at the moment
  # yay -S maturin
  # cd /tmp ; git clone https://github.com/nanoporetech/fast-ctc-decode.git
  # cd fast-ctc-decode ; make build

  subprocess.run([
    sys.executable,
    *('-m pip install --user --no-dependencies openvino-dev'.split(' '))
  ])
  import openvino


# Steal a utility file from https://github.com/geaxgx/depthai_blazepose/blob/main/mediapipe_utils.py
try:
  sys.path.append(os.path.abspath('build')) # assume build/mediapipe_utils.py exists, add build/ to search path
  import mediapipe_utils
except:
  traceback.print_exc()
  os.makedirs('build', exist_ok=True)
  subprocess.run([
    'wget', '-O', 'build/mediapipe_utils.py', 'https://raw.githubusercontent.com/geaxgx/depthai_blazepose/main/mediapipe_utils.py'
  ], check=True)
  import mediapipe_utils


def get_lan_ip():
  ip = ''
  try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = str(s.getsockname()[0])
    s.close()
  except:
    traceback.print_exc()
  return ip

async def http_index_req_handler(req):
  return aiohttp.web.FileResponse(path='index.html', status=200)

all_websockets = []
async def ws_req_handler(req):
  global all_websockets

  peername = req.transport.get_extra_info('peername')
  host_name = 'unk'
  if peername is not None:
    try:
      host_name, port = peername # only for ipv4 peers
    except:
      #traceback.print_exc()
      try:
        host_name, port, n1, n2 = peername # ipv6
      except:
        traceback.print_exc()
        host_name = str(peername)

  print('ws req from {}'.format(host_name))

  ws = aiohttp.web.WebSocketResponse()
  await ws.prepare(req)

  all_websockets.append(ws)

  try:
    async for msg in ws:
      if msg.type == aiohttp.WSMsgType.TEXT:
        
        print('WS From {}: {}'.format(host_name, msg.data))

      elif msg.type == aiohttp.WSMsgType.ERROR:
        print('ws connection closed with exception {}'.format(ws.exception()))
  except:
    traceback.print_exc()

  all_websockets.remove(ws)

  return ws

def frames(path):
    global exit_flag
    camera = cv2.VideoCapture(path)
    if not camera.isOpened():
        raise RuntimeError('Cannot open camera')

    while not exit_flag:
        _, img = camera.read()
        img = cv2.resize(img, (480, 320))
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+frame+b'\r\n'

def video_feed_gen(video_device='/dev/video0'):

  async def video_feed(request):
    response = aiohttp.web.StreamResponse()
    response.content_type = 'multipart/x-mixed-replace; boundary=frame'
    await response.prepare(request)

    for frame in frames(video_device):
      await response.write(frame)

    return response

  return video_feed


def dai_video_feed_gen():
  global exit_flag
  pipeline = depthai.Pipeline()

  # Define source and output
  camRgb = pipeline.createColorCamera()
  xoutRgb = pipeline.createXLinkOut()

  xoutRgb.setStreamName("rgb")

  # Properties
  camRgb.setPreviewSize(300, 300)
  camRgb.setInterleaved(False)
  camRgb.setColorOrder(depthai.ColorCameraProperties.ColorOrder.RGB)

  # Linking
  camRgb.preview.link(xoutRgb.input)

  async def video_feed(request):
    global exit_flag
    nonlocal pipeline

    response = aiohttp.web.StreamResponse()
    response.content_type = 'multipart/x-mixed-replace; boundary=frame'
    await response.prepare(request)

    with depthai.Device(pipeline) as device:

      print('Connected cameras: ', device.getConnectedCameras())
      # Print out usb speed
      print('Usb speed: ', device.getUsbSpeed().name)

      # Output queue will be used to get the rgb frames from the output defined above
      qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

      # for frame in frames(video_device):
      #     await response.write(frame)

      while not exit_flag:
        inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived
        cv_img = inRgb.getCvFrame()

        cv_img = cv2.resize(cv_img, (480, 320))
        frame = cv2.imencode('.jpg', cv_img)[1].tobytes()
        frame_packet = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+frame+b'\r\n'
        await response.write(frame_packet)

    return response

  return video_feed



def dai_depth_map():
  global exit_flag
  # Closer-in minimum depth, disparity range is doubled (from 95 to 190):
  extended_disparity = False
  # Better accuracy for longer distance, fractional disparity 32-levels:
  subpixel = False
  # Better handling for occlusions:
  lr_check = False

  # Create pipeline
  pipeline = depthai.Pipeline()

  # Define sources and outputs
  monoLeft = pipeline.createMonoCamera()
  monoRight = pipeline.createMonoCamera()
  depth = pipeline.createStereoDepth()
  xout = pipeline.createXLinkOut()

  xout.setStreamName("disparity")

  # Properties
  monoLeft.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_400_P)
  monoLeft.setBoardSocket(depthai.CameraBoardSocket.LEFT)
  monoRight.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_400_P)
  monoRight.setBoardSocket(depthai.CameraBoardSocket.RIGHT)

  # Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
  depth.initialConfig.setConfidenceThreshold(200)
  # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
  depth.initialConfig.setMedianFilter(depthai.MedianFilter.KERNEL_7x7)
  depth.setLeftRightCheck(lr_check)
  depth.setExtendedDisparity(extended_disparity)
  depth.setSubpixel(subpixel)

  # Linking
  monoLeft.out.link(depth.left)
  monoRight.out.link(depth.right)
  depth.disparity.link(xout.input)

  async def video_feed(request):
    global exit_flag
    nonlocal pipeline

    response = aiohttp.web.StreamResponse()
    response.content_type = 'multipart/x-mixed-replace; boundary=frame'
    await response.prepare(request)

    with depthai.Device(pipeline) as device:

      print('Connected cameras: ', device.getConnectedCameras())
      # Print out usb speed
      print('Usb speed: ', device.getUsbSpeed().name)

      # Output queue will be used to get the rgb frames from the output defined above
      qRgb = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)

      # for frame in frames(video_device):
      #     await response.write(frame)

      while not exit_flag:
        inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived
        cv_img = inRgb.getCvFrame()

        cv_img = cv2.resize(cv_img, (480, 320))
        frame = cv2.imencode('.jpg', cv_img)[1].tobytes()
        frame_packet = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+frame+b'\r\n'
        await response.write(frame_packet)

    return response

  return video_feed


def build_rgb_pose_manager_script(trace, pd_score_thresh, lm_score_thresh, force_detection, pad_h, img_h, frame_size, crop_w, rect_transf_scale, xyz, visibility_threshold):
    '''
    The code of the scripting node 'manager_script' depends on :
        - the NN model (full, lite, 831),
        - the score threshold,
        - the video frame shape
    So we build this code from the content of the file template_manager_script.py which is a python template
    '''
    # Read the template
    template_manager_script = 'build/template_manager_script.py'
    if not os.path.exists(template_manager_script):
      subprocess.run([
        'wget', '-O', template_manager_script, 'https://raw.githubusercontent.com/geaxgx/depthai_blazepose/a3ce15a25c82f4663e4d263c99d7f83ece59ab64/template_manager_script.py'
      ], check=True)
    with open(template_manager_script, 'r') as file:
        template = string.Template(file.read())
    
    # Perform the substitution
    code = template.substitute(
                _TRACE = "node.warn" if trace else "#",
                _pd_score_thresh = pd_score_thresh,
                _lm_score_thresh = lm_score_thresh,
                _force_detection = force_detection,
                _pad_h = pad_h,
                _img_h = img_h,
                _frame_size = frame_size,
                _crop_w = crop_w,
                _rect_transf_scale = rect_transf_scale,
                _IF_XYZ = "" if xyz else '"""',
                _buffer_size = 2910 if xyz else 2863,
                _visibility_threshold = visibility_threshold
    )
    # Remove comments and empty lines
    code = re.sub(r'"{3}.*?"{3}', '', code, flags=re.DOTALL)
    code = re.sub(r'#.*', '', code)
    code = re.sub('\n\s*\n', '\n', code)
    
    return code

def dai_rgb_pose():
  global exit_flag

  # Pipeline definition modified from https://github.com/geaxgx/depthai_blazepose/blob/main/BlazeposeDepthai.py#L259

  device = depthai.Device() # Just stays open

  pipeline = depthai.Pipeline()

  # Define source and output
  camRgb = pipeline.createColorCamera()
  xoutRgb = pipeline.createXLinkOut()

  xoutRgb.setStreamName("rgb")

  # Properties
  #camRgb.setPreviewSize(1920, 1080) # full resolution, super laggy given the inefficient JPEG encoding we currently use
  camRgb.setVideoSize(1920 // 4, 1080 // 4)
  camRgb.setPreviewSize(1920 // 4, 1080 // 4) # quarter resolution, feels real-time on my network

  frame_size, scale_nd =  mediapipe_utils.find_isp_scale_params(1920 // 4, is_height=False)
  camRgb.setIspScale(scale_nd[0], scale_nd[1])
  
  img_w = int(round(1080 * scale_nd[0] / scale_nd[1]))
  img_h = int(round(1920 * scale_nd[0] / scale_nd[1]))
  pad_h = (img_w - img_h) // 2
  pad_w = 0
  crop_w = 0

  nb_kps = 33 # where is this used?
  pd_score_thresh = 0.5
  lm_score_thresh = 0.7
  force_detection = False # when true ignore previous frame person data
  rect_transf_scale = 1.25
  visibility_threshold = 0.5
  internal_fps = 8 # required by full model

  pd_input_length = 224
  lm_input_length = 256

  camRgb.setFps(internal_fps)
  camRgb.setBoardSocket(depthai.CameraBoardSocket.RGB)

  camRgb.setInterleaved(False)
  camRgb.setColorOrder(depthai.ColorCameraProperties.ColorOrder.RGB)

  manager_script = pipeline.create(depthai.node.Script)
  manager_script.setScript(build_rgb_pose_manager_script(
    trace=False, # no debug data for now
    pd_score_thresh=pd_score_thresh,
    lm_score_thresh=lm_score_thresh,
    force_detection=force_detection,
    pad_h=pad_h,
    img_h=img_h,
    frame_size=frame_size,
    crop_w=crop_w,
    rect_transf_scale=rect_transf_scale,
    xyz=True, # pull depth data for x,y locations
    visibility_threshold=visibility_threshold
  ))

  # For now, RGB needs fixed focus to properly align with depth.
  # The value used during calibration should be used here
  calib_data = device.readCalibration()
  calib_lens_pos = calib_data.getLensPosition(depthai.CameraBoardSocket.RGB)
  print(f"RGB calibration lens position: {calib_lens_pos}")
  camRgb.initialControl.setManualFocus(calib_lens_pos)

  mono_resolution = depthai.MonoCameraProperties.SensorResolution.THE_400_P
  left = pipeline.createMonoCamera()
  left.setBoardSocket(depthai.CameraBoardSocket.LEFT)
  left.setResolution(mono_resolution)
  left.setFps(internal_fps)

  right = pipeline.createMonoCamera()
  right.setBoardSocket(depthai.CameraBoardSocket.RIGHT)
  right.setResolution(mono_resolution)
  right.setFps(internal_fps)

  stereo = pipeline.createStereoDepth()
  stereo.initialConfig.setConfidenceThreshold(230)
  # LR-check is required for depth alignment
  stereo.setLeftRightCheck(True)
  stereo.setDepthAlign(depthai.CameraBoardSocket.RGB)
  stereo.setSubpixel(False)  # subpixel True -> latency
  # MEDIAN_OFF necessary in depthai 2.7.2. 
  # Otherwise : [critical] Fatal error. Please report to developers. Log: 'StereoSipp' '533'
  # stereo.setMedianFilter(dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF)

  spatial_location_calculator = pipeline.createSpatialLocationCalculator()
  spatial_location_calculator.inputConfig.setWaitForMessage(True)
  spatial_location_calculator.inputDepth.setBlocking(False)
  spatial_location_calculator.inputDepth.setQueueSize(1)

  left.out.link(stereo.left)
  right.out.link(stereo.right)

  stereo.depth.link(spatial_location_calculator.inputDepth)


  manager_script.outputs['spatial_location_config'].link(spatial_location_calculator.inputConfig)
  spatial_location_calculator.out.link(manager_script.inputs['spatial_data'])

  # Define pose detection pre processing (resize preview to (self.pd_input_length, self.pd_input_length))
  print("Creating Pose Detection pre processing image manip...")
  pre_pd_manip = pipeline.create(depthai.node.ImageManip)
  pre_pd_manip.setMaxOutputFrameSize(pd_input_length*pd_input_length*3)
  pre_pd_manip.inputConfig.setWaitForMessage(True)
  pre_pd_manip.inputImage.setQueueSize(1)
  pre_pd_manip.inputImage.setBlocking(False)
  camRgb.preview.link(pre_pd_manip.inputImage)
  manager_script.outputs['pre_pd_manip_cfg'].link(pre_pd_manip.inputConfig)

  # Linking
  camRgb.preview.link(xoutRgb.input)

  async def video_feed(request):
    nonlocal pipeline
    global exit_flag

    response = aiohttp.web.StreamResponse()
    response.content_type = 'multipart/x-mixed-replace; boundary=frame'
    await response.prepare(request)

    print('Connected cameras: ', device.getConnectedCameras())
    # Print out usb speed
    print('Usb speed: ', device.getUsbSpeed().name)

    device.startPipeline(pipeline)

    # Output queue will be used to get the rgb frames from the output defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while not exit_flag:
      inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived
      cv_img = inRgb.getCvFrame()

      #cv_img = cv2.resize(cv_img, (480, 320))
      frame = cv2.imencode('.jpg', cv_img)[1].tobytes()
      frame_packet = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+frame+b'\r\n'
      await response.write(frame_packet)

    return response

  return video_feed



exit_flag = False # use in infinite loops to make them slightly less infinite
def on_signal():
  global exit_flag
  exit_flag = True
  print('Exiting...')
  sys.exit(1)

async def start_background_tasks(arg):
  loop = asyncio.get_event_loop()
  for sig_num in [signal.SIGINT, signal.SIGTERM]:
    loop.add_signal_handler(sig_num, on_signal)


async def stop_background_tasks(arg):
  pass
  

def main(args=sys.argv):
  # Ensure always at repo root
  os.chdir( os.path.dirname(os.path.abspath(__file__)) )

  print(f'args={args}')

  # Ensure we can dump things into the .gitignored models/ dir
  os.makedirs('models', exist_ok=True)

  # Ensure we have models we use
  # subprocess.run([
  #   'omz_downloader', '-o', 'models/', '--name', ''
  # ])
  files_and_urls = [
    ('models/pose_landmark_full_FP32.xml', 'https://raw.githubusercontent.com/geaxgx/openvino_blazepose/main/models/pose_landmark_full_FP32.xml'),
    ('models/pose_landmark_full_FP32.bin', 'https://raw.githubusercontent.com/geaxgx/openvino_blazepose/main/models/pose_landmark_full_FP32.bin'),
  ]
  for file, url in files_and_urls:
    if not os.path.exists(file):
      print(f'Downloading {file} from {url} using wget')
      subprocess.run([
        'wget', '-O', file, url
      ], check=True)

  http_port = 8000

  server = aiohttp.web.Application()

  video_feeds = []
  for i in range(0, 9):
    v_dev = f'/dev/video{i}'
    if os.path.exists(v_dev):
      video_feeds.append(
        aiohttp.web.get(f'/video{i}', video_feed_gen(v_dev))
      )
      print(f'Serving {v_dev} at /video{i}')

  if len(depthai.Device.getAllAvailableDevices()) > 0:
    video_feeds.append(
      aiohttp.web.get(f'/depthai', dai_video_feed_gen())
    )
    print(f'Serving  /depthai')

    video_feeds.append(
      aiohttp.web.get(f'/depth_map', dai_depth_map())
    )
    print(f'Serving  /depth_map')

    video_feeds.append(
      aiohttp.web.get(f'/rgb_pose', dai_rgb_pose())
    )
    print(f'Serving  /rgb_pose')

  server.add_routes([
    aiohttp.web.get('/', http_index_req_handler),
    aiohttp.web.get('/ws', ws_req_handler),
    #aiohttp.web.get('/video', video_feed),
    *video_feeds,
    
  ])

  server.on_startup.append(start_background_tasks)
  server.on_shutdown.append(stop_background_tasks)

  my_lan_ip = get_lan_ip()
  print()
  print(f'Listening on http://0.0.0.0:{http_port}/')
  print(f'Listening on http://{my_lan_ip}:{http_port}/')
  print()
  for vf in video_feeds:
    print(f'Useful link: http://{my_lan_ip}:{http_port}{vf.path}')
  print()
  aiohttp.web.run_app(server, port=http_port)


if __name__ == '__main__':
  main()

