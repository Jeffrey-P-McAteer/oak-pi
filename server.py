#!/usr/bin/env python

import os
import sys
import subprocess
import traceback
import socket

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

  server.add_routes([
    aiohttp.web.get('/', http_index_req_handler),
    aiohttp.web.get('/ws', ws_req_handler),
    #aiohttp.web.get('/video', video_feed),
    *video_feeds,
    
  ])

  #server.on_startup.append(start_background_tasks)
  #server.on_shutdown.append(stop_background_tasks)

  print()
  print(f'Listening on http://0.0.0.0:{http_port}/')
  print(f'Listening on http://{get_lan_ip()}:{http_port}/')
  print()
  aiohttp.web.run_app(server, port=http_port)


if __name__ == '__main__':
  main()

