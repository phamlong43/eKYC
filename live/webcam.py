import argparse
import asyncio
import logging
import ssl
import struct
import socket
import time
from fractions import Fraction
import aiohttp_cors
import cv2
import numpy as np
import torch
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc import VideoStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaRelay
from aiortc.rtcrtpsender import RTCRtpSender
from av import VideoFrame
from facenet_pytorch import InceptionResnetV1
from facenet_pytorch import MTCNN
from pymongo import MongoClient
from PIL import Image
import torch.nn.functional as F
import requests
import json
from datetime import datetime
import os

from VideoWorker import process_frames, cleanup
import schedule_service
import subprocess

async def call_curl():
    command = ['curl', '-X', 'POST', 'http://raspberrypi.local:8005/led/2']
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        print("Status Code:", result.returncode)
    except Exception as e:
        print(f"Error: {e}")

user_url = 'http://localhost:8080/attend'
client = MongoClient("mongodb://localhost:27017/")
db = client["embedding"]
collection = db["user_embedding"]
user_objects = []
embeddings = []
documents = collection.find()
score_timers = {}
relay = None
webcam = None
server_port = 8554
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', server_port))
server_socket.listen(1)
print("Waiting for connection...")
conn, addr = server_socket.accept()
print(f"Connection established with {addr}")
MAX_FRAME_SIZE = 10 * 1024 * 1024
model = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(keep_all=False, device='cpu')
image_base_path = 'images/'
response_value=0

def get_all():
    for document in documents:
        user_id = document['_id']
        user_name = document['userName']
        embedding_list = document['embedding']
        embedding_np = torch.tensor(embedding_list).reshape(1, -1)
        user = {
            'id': user_id,
            'userName': user_name,
        }
        user_objects.append(user)
        embeddings.append(embedding_np)


def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    face = mtcnn(image)
    face = face.unsqueeze(0)
    embedding = model(face)
    return embedding


def cosine_similarity(embedding1, embedding2):
    return F.cosine_similarity(embedding1, embedding2)


def recognize_faces(frame_embedding, db_embeddings, users, threshold=0.7):
    similarities = []
    for db_embedding in db_embeddings:
        if db_embedding.shape != frame_embedding.shape:
            continue
        sim = cosine_similarity(frame_embedding, db_embedding)
        similarities.append(sim.item())

    relate = max(similarities)
    index = similarities.index(relate)
    if relate > threshold:
        print("User match: ", users[index], " with similarity ", relate)
        asyncio.create_task(call_curl())
        return users[index], similarities[index]

    return "Unknown", None


def get_today_folder():
    today_date = datetime.now().strftime("%d-%m-%Y")
    folder_path = os.path.join(image_base_path, today_date)
    return folder_path


async def check_exist(img, user):
    folder_path = get_today_folder()
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    userName = user["userName"]
    user_id = user["id"]
    userName = userName.replace(' ', '_')
    file_name = f"{userName}-{user_id}.jpg"
    file_path = os.path.join(folder_path, file_name)
    if not os.path.exists(file_path):
        cv2.imwrite(file_path, img)
        await handle_recognition(user)
        print(f"Image saved to: {file_path}")
    else:
        print(f"{file_name} already exists")


async def handle_recognition(user):
    state = schedule_service.check_attendance_status()
    if not state["status"]:
        print("Attendance status is unknown")
        return
    print("making attend request with user: ", user["userName"], " state: ", state["status"])
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "userId": user["id"],
        "date": datetime.now().astimezone().isoformat(),
        "state": state["status"],
        "isAuto": True
    }
    try:
        response = requests.post(user_url, data=json.dumps(data), headers=headers)
        if response.status_code == 200:
            print("Success:", response.json())
        else:
            print(f"Failed with status code {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


def transform_to_tensor(face):
    face = cv2.resize(face, (160, 160))
    # Converts the resized face (which is a NumPy array, typical of OpenCV images)
    # into a PyTorch tensor. PyTorch models operate on tensors,
    # so this step makes the image compatible with PyTorch.
    # OpenCV represents images in the format (height, width, channels) (e.g., (160, 160, 3) for RGB images).
    # PyTorch models typically expect tensors in the format (channels, height, width) (e.g., (3, 160, 160) for RGB).
    face = torch.tensor(face, dtype=torch.float32).permute(2, 0, 1)
    # Pixel values in OpenCV are usually integers in the range [0, 255].
    # Neural networks often perform better when inputs are normalized,
    # as it helps stabilize the learning process.
    face = face / 255.0
    return face


async def process_frame(img, face_boxes,conn):
    if face_boxes is None or len(face_boxes) == 0:
        print("No faces detected")
        return

    for face_bbox in face_boxes:
        if face_bbox is None or len(face_bbox) != 4:
            print("Invalid face bounding box")
            continue

        x1, y1, x2, y2 = [max(0, int(coord)) for coord in face_bbox]

        # Ensure coordinates are within image dimensions
        height, width = img.shape[:2]
        x1 = min(x1, width - 1)
        x2 = min(x2, width - 1)
        y1 = min(y1, height - 1)
        y2 = min(y2, height - 1)

        # Check if we have a valid face region
        if x2 <= x1 or y2 <= y1:
            print("Invalid face region dimensions")
            continue

        try:
            face = img[y1:y2, x1:x2]  # Crop the face from the image
            face_tensor = transform_to_tensor(face)
            face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension

            # Compute face embedding
            embedding = model(face_tensor)

            if embedding is not None:
                # Recognize face
                user, score = recognize_faces(embedding, embeddings, user_objects)
                if score:
                    user_id = user["userName"]
                    current_time = time.time()

                    if user_id in score_timers:
                        elapsed_time = current_time - score_timers[user_id]
                        if elapsed_time >= 1:
                            await check_exist(img, user)
                            del score_timers[user_id]
                    else:
                        score_timers[user_id] = current_time
        except Exception as e:
            print(f"Error processing face: {str(e)}")
            continue


def _create_default_frame():

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Có thể thêm text vào frame
    cv2.putText(frame, 'No Signal', (220, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame


class SocketVideoStreamTrack(VideoStreamTrack):
    def __init__(self, conn, buffer_size=8192, step=2):
        super().__init__()
        self.conn = conn
        self.start_time = time.time()
        self.frame_counter = 0
        self.buffer_size = buffer_size
        self.step = step
        self.default_frame = _create_default_frame()

    async def recv(self):
        try:
            loop = asyncio.get_event_loop()
            frame_size = await loop.run_in_executor(None, self._recv_frame_size)
            if frame_size <= 0 or frame_size > MAX_FRAME_SIZE:
                print(f"Frame size {frame_size} exceeds the maximum allowed size, skipping this frame.")
                frame = VideoFrame.from_ndarray(self.default_frame, format="bgr24")
                self.frame_counter += 1
                frame.pts = self.frame_counter
                frame.time_base = Fraction(1, 30)
                return frame
            frame_data = await loop.run_in_executor(None, self._recv_frame_data, frame_size)
            if frame_data is None:
                frame = VideoFrame.from_ndarray(self.default_frame, format="bgr24")
                return frame
            img = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode frame")
            img = cv2.flip(img, 1)
            img = cv2.resize(img, (640, 480))
            asyncio.create_task(process_frames(img))
            boxes, _ = mtcnn.detect(img)
            response_value = 0
            if boxes is not None:
                response_value = 1
                for box in boxes:
                    bbox = list(map(int, box.tolist()))
                    if self.frame_counter % self.step == 0:
                        asyncio.create_task(process_frame(img, boxes, self.conn))
                    img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 6)
            conn.sendall(response_value.to_bytes(1, byteorder='big'))
            frame = VideoFrame.from_ndarray(img, format="bgr24")
            self.frame_counter += 1
            frame.pts = self.frame_counter
            frame.time_base = Fraction(1, 30)
            return frame

        except Exception as e:
            print(f"Error receiving frame: {e}")
            raise


    def _recv_frame_size(self):
        size_data = b''
        while len(size_data) < 8:
            size_data += self.conn.recv(8 - len(size_data))
        frame_size = struct.unpack("L", size_data)[0]
        return frame_size

    def _recv_frame_data(self, frame_size):
        frame_data = b''
        while len(frame_data) < frame_size:
            frame_data += self.conn.recv(frame_size - len(frame_data))
        return frame_data


def create_local_tracks(play_from, decode, socket_conn=None):
    global relay

    if play_from:
        player = MediaPlayer(play_from, decode=decode)
        return player.audio, player.video
    elif socket_conn:
        video_track = SocketVideoStreamTrack(socket_conn)
        return None, video_track
    else:
        options = {"framerate": "30", "video_size": "640x480"}
        if relay is None:
            webcam = MediaPlayer("/dev/video0", format="v4l2", options=options)
            relay = MediaRelay()
        video_track = relay.subscribe(webcam.video)
        return None, video_track


def force_codec(pc, sender, forced_codec):
    kind = forced_codec.split("/")[0]
    codecs = RTCRtpSender.getCapabilities(kind).codecs
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    transceiver.setCodecPreferences(
        [codec for codec in codecs if codec.mimeType == forced_codec]
    )


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
        if pc.connectionState == "closed":
            cleanup()

    # open media source
    audio, video = create_local_tracks(
        args.play_from, decode=not args.play_without_decoding,
        socket_conn=conn
    )

    if video:
        video_sender = pc.addTrack(video)
        if args.video_codec:
            force_codec(pc, video_sender, args.video_codec)
        elif args.play_without_decoding:
            raise Exception("You must specify the video codec using --video-codec")

    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


pcs = set()


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    get_all()
    parser = argparse.ArgumentParser(description="WebRTC webcam demo")
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument("--play-from", help="Read the media from a file and sent it.")
    parser.add_argument(
        "--play-without-decoding",
        help=(
            "Read the media without decoding it (experimental). "
            "For now it only works with an MPEGTS container with only H.264 video."
        ),
        action="store_true",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8081, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    parser.add_argument(
        "--audio-codec", help="Force a specific audio codec (e.g. audio/opus)"
    )
    parser.add_argument(
        "--video-codec", help="Force a specific video codec (e.g. video/H264)"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })
    app.on_shutdown.append(on_shutdown)
    app.router.add_post("/offer", offer)
    for route in list(app.router.routes()):
        cors.add(route)
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)
