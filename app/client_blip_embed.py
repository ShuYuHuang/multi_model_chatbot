import requests
import asyncio
from time import time
from typing import Any, Optional, Dict, List

import cv2
import numpy as np


ENCODE_PARAM = [int(cv2.IMWRITE_JPEG_QUALITY), 95]  # Adjust the quality as needed
BLIP_URL = 'http://10.100.100.106:10002'
DEFAULT_VQA_PROMPT = 'Describe the image concisely.'

def convert_frame_to_base64(frame):
    """Converts a frame to base64 format.
    """
    # Get the frame bytes.
    retval, buffer = cv2.imencode(".jpg", frame, ENCODE_PARAM)

    # Encode the frame bytes to base64.
    frame_base64 = base64.b64encode(buffer)

    # Return the base64 encoded frame.
    return frame_base64

def vframe_to_base64(
        video_file,
        skip_frames_interval=0.2): # set interval between samples in seconds
    cap = cv2.VideoCapture(video_file)

    # Get basic video properties
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / frame_rate)

    num_frames_to_skip = int(skip_frames_interval * frame_rate) # Calculate the number of frames to skip.
    skip_frames_counter = 0 # Initialize a counter to track the number of frames skipped.

    # Convert the video frames to base64.
    frames_64 = []
#     frames = []
#     frames_b = []
    while True:
        if skip_frames_counter < num_frames_to_skip:
            skip_frames_counter += 1
            cap.read()
            continue
        skip_frames_counter = 0
        # Read the current frame from the video.
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_base64 = convert_frame_to_base64(frame)
        frames_64.append(frame_base64)
        # frames.append(frame)
        # frames_b.append(frame_bytes)
    cap.release()
    return frames_64, frame_rate, video_length

class Blip2Captioner:
    def __init__(self, url: str,
                 task: str = "image_caption"):
        self.url = url
        self.task = task
    
    
    # Embedding for texts
    def load_model(self) -> Dict:
        return requests.post(self.url + "/load_model", data={"task": self.task}, timeout=300).json()
    
    async def send_frames(self, frame64, prompt = DEFAULT_VQA_PROMPT):
        if self.task == "image_caption":
           return requests.post(self.url + "/" + self.task, data={'img_base64': frame64}, timeout=30).json()["caption"][0]
        elif self.task == "vqa":
           pload = {
                "img_base64": frame64,
                "prompt": prompt
            }
           return requests.post(self.url + "/" + self.task, data=pload, timeout=30).json()["answer"][0]
           
    async def send_frame_parallel(self, client_frames_64: bytes, prompt=DEFAULT_VQA_PROMPT) -> Dict:
        tasks = [self.send_frames(frame64, prompt) for frame64 in client_frames_64]
        responses = await asyncio.gather(*tasks)
        return responses

    def caption_video(self, video_file: str, skip_frames_interval: float = 0.2, prompt=DEFAULT_VQA_PROMPT) -> Dict:
        client_frames_64, frame_rate, video_length = vframe_to_base64(video_file, skip_frames_interval)

        loop = asyncio.get_event_loop()
        start_time = time()
        responses = loop.run_until_complete(self.send_frame_parallel(client_frames_64, prompt))
        time_cost = time() - start_time
        
        print(f"Length of film: {video_length}s, FPS: {frame_rate}")
        print(f"Processed {len(responses)} frames with {time_cost:.02f}s , with {time_cost/len(responses):.02f}s per frame")
       
        return [(sec, resp) for sec, resp in zip(np.linspace(1/frame_rate, video_length, len(client_frames_64)), responses)]



if __name__=="__main__":
    import base64

    BLIP_URL = "http://10.100.100.106:10002"
    LLAVA_URL1 = 'http://10.100.100.106:8015/image_chat'

    test_file = "../../video_try/output_video.mp4"

    captioner = Blip2Captioner(LLAVA_URL1)
    captioner = Blip2Captioner(BLIP_URL, "image_caption")
    captioner.load_model()
    
    responses = captioner.caption_video(test_file, skip_frames_interval = .5)

    _ = [print(r) for r in responses]