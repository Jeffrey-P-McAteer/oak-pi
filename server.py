#!/usr/bin/env python

import os
import sys
import subprocess
import traceback
import socket
import platform
import signal
import asyncio

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

# python -m pip install --user openvino # intel's model stuff, mostly useful because it provides omz_downloader
# try:
#   import openvino
# except:
#   traceback.print_exc()
#   subprocess.run([
#     sys.executable,
#     *('-m pip install --user openvino'.split(' '))
#   ])
#   import openvino




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
    camera = cv2.VideoCapture(path)
    if not camera.isOpened():
        raise RuntimeError('Cannot open camera')

    while True:
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

      while True:
        inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived
        cv_img = inRgb.getCvFrame()

        cv_img = cv2.resize(cv_img, (480, 320))
        frame = cv2.imencode('.jpg', cv_img)[1].tobytes()
        frame_packet = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+frame+b'\r\n'
        await response.write(frame_packet)

    return response

  return video_feed



def dai_depth_map():
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

      while True:
        inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived
        cv_img = inRgb.getCvFrame()

        cv_img = cv2.resize(cv_img, (480, 320))
        frame = cv2.imencode('.jpg', cv_img)[1].tobytes()
        frame_packet = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+frame+b'\r\n'
        await response.write(frame_packet)

    return response

  return video_feed


def dai_rgb_pose():
  pipeline = depthai.Pipeline()

  # Define source and output
  camRgb = pipeline.createColorCamera()
  xoutRgb = pipeline.createXLinkOut()

  xoutRgb.setStreamName("rgb")

  # Properties
  camRgb.setPreviewSize(4208, 3120) # full resolution
  #camRgb.setPreviewSize(4208 // 4, 3120 // 4) # quarter resolution
  camRgb.setInterleaved(False)
  camRgb.setColorOrder(depthai.ColorCameraProperties.ColorOrder.RGB)

  # Linking
  camRgb.preview.link(xoutRgb.input)

  async def video_feed(request):
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

      while True:
        inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived
        cv_img = inRgb.getCvFrame()

        #cv_img = cv2.resize(cv_img, (480, 320))
        frame = cv2.imencode('.jpg', cv_img)[1].tobytes()
        frame_packet = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+frame+b'\r\n'
        await response.write(frame_packet)

    return response

  return video_feed



def on_signal():
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

  print()
  print(f'Listening on http://0.0.0.0:{http_port}/')
  print(f'Listening on http://{get_lan_ip()}:{http_port}/')
  print()
  aiohttp.web.run_app(server, port=http_port)


if __name__ == '__main__':
  main()

