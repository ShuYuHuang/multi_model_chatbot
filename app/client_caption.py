from os import path
from time import time
from typing import Any, Optional, Dict, List
import requests
import asyncio
import base64

import cv2
import numpy as np

from params import DEFAULT_TMP_DIR


ENCODE_PARAM = [int(cv2.IMWRITE_JPEG_QUALITY), 95]  # Adjust the quality as needed
BLIP_URL = 'http://10.100.100.106:10002'
DEFAULT_VQA_PROMPT = 'Describe the image concisely.'
DEFAULT_LLAVA_POMPT = 'Describe the image concisely in 20 words.'

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
        skip_frames_interval=0.2
    ): # set interval between samples in seconds
    cap = cv2.VideoCapture(video_file)

    # Get basic video properties
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / frame_rate)

    num_frames_to_skip = int(skip_frames_interval * frame_rate) # Calculate the number of frames to skip.
    skip_frames_counter = 0 # Initialize a counter to track the number of frames skipped.

    # Convert the video frames to base64.
    frames_64 = []
    frames = []
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

        frame_ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_base64 = convert_frame_to_base64(frame_)
        frames_64.append(frame_base64)
        frames.append(frame)
        # frames_b.append(frame_bytes)
    cap.release()
    return frames_64, frames, frame_rate, video_length

class BaseCaptioner:
    def __init__(self, url, *args,**kwargs):
        
        pass
    async def send_frames(self, frame64, prompt):
        """Describe the function to send frame to captioning server"""
        pass

    async def send_frame_parallel(self, client_frames_64: bytes, prompt=DEFAULT_VQA_PROMPT) -> Dict:
        tasks = [self.send_frames(frame64, prompt) for frame64 in client_frames_64]
        responses = await asyncio.gather(*tasks)
        return responses

    def caption_video(
            self,
            video_file: str,
            skip_frames_interval: float = 0.2,
            prompt: str = None,
            tmp_dir: str = None,
            ) -> Dict:
        client_frames_64, client_frames, frame_rate, video_length = vframe_to_base64(video_file, skip_frames_interval)

        # Save tmp files
        if tmp_dir:
            vfilename = path.splitext(path.basename(video_file))
            for i, frame in enumerate(client_frames):
                tmp_filename = path.join(tmp_dir, f"{vfilename}_frame_{i}.jpg")
                cv2.imwrite(tmp_filename, frame, ENCODE_PARAM)

        loop = asyncio.get_event_loop()
        start_time = time()
        if prompt:
            # use assigned prompt
            responses = loop.run_until_complete(self.send_frame_parallel(client_frames_64, prompt))
        else:
            # use default prompt
            responses = loop.run_until_complete(self.send_frame_parallel(client_frames_64))

        time_cost = time() - start_time

        print(f"Length of film: {video_length}s, FPS: {frame_rate}, new FPS: {1/(skip_frames_interval+1e-10):.04f}")

        print(f"Processed {len(responses)} frames with {time_cost:.02f}s , with {time_cost/len(responses):.02f}s per frame")

        return [(sec, resp) for sec, resp in zip(np.linspace(1/frame_rate, video_length, len(client_frames_64)), responses)]


class Blip2Captioner(BaseCaptioner):
    def __init__(self, url: str,
                 task: str = "image_caption"):
        self.url = url
        self.task = task
    
    # Embedding for texts
    def load_model(self) -> Dict:
        return requests.post(self.url + "/load_model", data={"task": self.task}, timeout=300).json()
    
    async def send_frames(self, frame64, prompt = DEFAULT_LLAVA_POMPT):
        if self.task == "image_caption":
            pload = {
                "img_base64": frame64,
                "prompt": f"Question: {DEFAULT_VQA_PROMPT} Answer:"
            }
            return requests.post(self.url + "/" + self.task, data=pload, timeout=30).json()["caption"][0]
        elif self.task == "vqa":
           pload = {
                "img_base64": frame64,
                "prompt": prompt
            }
           return requests.post(self.url + "/" + self.task, data=pload, timeout=30).json()["answer"][0]
           
class LlavaCaptioner(BaseCaptioner):
    def __init__(self, url: str):
        self.url = url
    
    # Embedding for texts
    # def load_model(self) -> Dict:
    #     return requests.post(self.url + "/load_model", data={"task": self.task}, timeout=300).json()
    
    async def send_frames(self, frame64, prompt = DEFAULT_LLAVA_POMPT):
        pload = {
            "img_base64": frame64,
            "query": f"Question: {prompt} Answer:"
        }
        return await requests.post(self.url + "/image_chat", data=pload, timeout=30).json()['assistant']


if __name__=="__main__":


    BLIP_URL = "http://10.100.100.106:10002"
    LLAVA_URL1 = 'http://10.100.100.106:8015'

    
    test_file = "../../video_try/output_video.mp4"

    captioner = LlavaCaptioner(LLAVA_URL1)

    # captioner = Blip2Captioner(BLIP_URL, "vqa")
    # captioner.load_model()
    
    responses = captioner.caption_video(
        test_file,
        skip_frames_interval = .2,
        tmp_dir = DEFAULT_TMP_DIR
        )

    _ = [print(r) for r in responses]



   