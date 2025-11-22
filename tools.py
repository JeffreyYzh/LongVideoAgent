from qwen_agent.tools.base import BaseTool, register_tool
import cv2
import os
import glob
from sys_prompt import GLOBAL_REFINER_PROMPT, SEGMENT_REFINER_PROMPT
from qwen_agent.agents import Assistant
from qwen_agent.utils.output_beautify import typewriter_print
from moviepy import VideoFileClip, concatenate_videoclips
from env import SAVE_ROOT, MODEL_NAME, API_KEY

@register_tool("extract_frames_by_index")
class ExtractFramesByIndexTool(BaseTool):
    description = "given the input video path, extract the frames accoring to the provided image indexs, and output the directory of the extracted frames"
    parameters = [
        {
            "name": "video_path",
            "type": "string",
            "description": "input video path"
        },
        {
            "name": "frame_indices",
            "type": "array",
            "items": {"type": "number"},
            "description": "the indices of frames to be extracted"
        },
    ]

    def call(self, params: dict, **kwargs):
        params = eval(params)
        video_path = params["video_path"]
        frame_indices = params["frame_indices"]
        out_dir = SAVE_ROOT

        os.makedirs(out_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Can not open the video {video_path}")

        out_path_list = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            if not success:
                print(f"Can not read {idx} frame, maybe exceed the frame range.")
                continue
            
            frame = cv2.resize(frame, (1280, 720))

            out_path = os.path.join(out_dir, f"{idx}.png")
            cv2.imwrite(out_path, frame)
            print(f"Saved to {out_path}")
            out_path_list.append(out_path)
        cap.release()
        return {"save_path_list": out_path_list}

@register_tool("concat_videos_timewise")
class ConcatVideosTimewiseTool(BaseTool):
    description = "Concat all the video along the time axis in the directory"
    parameters = [
        {
            "name": "input_dir",
            "type": "string",
            "description": "the directory to be operated"
        }
    ]

    def call(self, params: dict, **kwargs):
        params = eval(params)
        video_dir = params["input_dir"]

        files = [x for x in os.listdir(video_dir) if x.endswith('mp4') and not x.split('/')[-1].startswith('global')]   
        files = sorted(files) 
        output_path = os.path.join(video_dir, 'final_video.mp4')
        print(files)

        clips = []
        try:
            for p in files:
                p = os.path.join(video_dir, p)
                clips.append(VideoFileClip(str(p)))

            fps = 16
            final = concatenate_videoclips(clips, method="compose")
            final.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="aac",
                fps=fps,
                ffmpeg_params=[
                    "-pix_fmt", "yuv420p",     
                    "-movflags", "+faststart" 
                ],
            )
        finally:
            for c in clips:
                try:
                    c.close()
                except Exception:
                    pass


        print(f"[OK] The output video has been saved to: {output_path}")
        return {"save_dir": video_dir}


@register_tool("refine_prompts")
class RefinePrompt(BaseTool):
    description = "Refine the provided prompt, give more details."
    parameters = [
        {
            "name": "prompt",
            "type": "string",
            "description": "input concise prompt"
        },
        {
            "name": "is_global",
            "type": "bool",
            "description": "whether it is a global prompt refiner, use True or False to indicate"
        },
        {
            "name": "number",
            "type": "int",
            "default": 1,
            "description": "if it is a segment prompt refiner, this is the number of prompt refiner should generate."
        }
    ]

    def call(self, params: dict, **kwargs):
        params = params.replace('true', 'True')
        params = params.replace('false', 'False')
        llm_cfg = {
            'model': MODEL_NAME,
            'model_type': 'qwen_dashscope',
            'api_key': API_KEY,
            'generate_cfg': {
                'top_p': 0.8
            }
        }
        params = eval(params)
        prompt = params["prompt"]
        is_global = params["is_global"]

        if is_global:
            bot = Assistant(llm=llm_cfg,
                system_message=GLOBAL_REFINER_PROMPT,
                )
            query = f'Please refine the global prompt: {prompt}'
            messages = []
            messages.append({'role': 'user', 'content': query})
            response_plain_text = ''
            for response in bot.run(messages=messages):
                response_plain_text = typewriter_print(response, response_plain_text)
            prompt_list = response[0]['content']
        else:
            bot = Assistant(llm=llm_cfg,
                system_message=SEGMENT_REFINER_PROMPT,
                )
            number = params['number']
            query = f'Please give {number} segmented prompts based on the global prompt: {prompt}'
            messages = []
            messages.append({'role': 'user', 'content': query})
            response_plain_text = ''
            for response in bot.run(messages=messages):
                response_plain_text = typewriter_print(response, response_plain_text)            
            prompt_list = response[0]['content']

        return {"refine_prompts": prompt_list}

@register_tool("calculate_frame_index")
class Frame_index_calculator(BaseTool):
    description = "Given the video length (second), this function should output the proper number of segments and extracted image index of the video"
    parameters = [
        {
            "name": "video_length",
            "default": 10,
            "type": "int",
            "description": "the length of the video (second)"
        }
    ]

    def call(self, params: dict, **kwargs):
        params = eval(params)
        video_length = params["video_length"]
        length_per_video = 5
        frames_per_video = 81   # Wan 2.1

        num_segments = video_length // length_per_video
        segment_length = frames_per_video // num_segments
        image_indexs = [x*segment_length if x*segment_length < frames_per_video - 1 else frames_per_video - 1 for x in range(num_segments+1)]

        return {"num_segments": num_segments,
                "frame_indexs": image_indexs}