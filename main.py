import urllib.parse
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.utils.output_beautify import typewriter_print
from qwen_agent.llm import get_chat_model
import datetime
import argparse
import logging
import os
import sys
import warnings
import json
from datetime import datetime
import random
import torch
import torch.distributed as dist
from PIL import Image
import cv2
import glob
warnings.filterwarnings('ignore')

from sys_prompt import SYS_PROMPT, SEGMENT_REFINER_PROMPT, GLOBAL_REFINER_PROMPT
from tools import *
from video_gen import GenerateT2VTool, GenerateFLF2VTool
from env import API_KEY, SAVE_ROOT, MODEL_NAME


llm_cfg = {
    'model': MODEL_NAME,
    'model_type': 'qwen_dashscope',
    'api_key': API_KEY,  
    'generate_cfg': {
        'top_p': 0.9
    }
}
tools = ['extract_frames_by_index', 'concat_videos_timewise', 'calculate_frame_index', 'generate_flf2v', 'generate_t2v', 'refine_prompts']


system_instruction = SYS_PROMPT

bot = Assistant(llm=llm_cfg,
                system_message=system_instruction,
                function_list=tools,
            
                )


prompt = 'Please help me generate a 10 second video. The prompt is: A dog happily runs to a man.'

messages = [{'role': 'user', 'content': prompt}]
response = []
response_plain_text = ''
for response in bot.run(messages=messages):
    response_plain_text = typewriter_print(response, response_plain_text)
messages.extend(response)