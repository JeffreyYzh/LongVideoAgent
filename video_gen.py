# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random
import gc
import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_image, cache_video, str2bool
from generate import EXAMPLE_PROMPT
from qwen_agent.tools.base import BaseTool, register_tool
from env import SAVE_ROOT, T2V_MODEL_CKPT, FLF2V_MODEL_CKPT


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 50

    if args.sample_shift is None:
        args.sample_shift = 5.0
    elif "flf2v" in args.task or "vace" in args.task:
        args.sample_shift = 16

    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from a image or video. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--src_video",
        type=str,
        default=None,
        help="The file of the source video. Default None.")
    parser.add_argument(
        "--src_mask",
        type=str,
        default=None,
        help="The file of the source mask. Default None.")
    parser.add_argument(
        "--src_ref_images",
        type=str,
        default=None,
        help="The file list of the source reference images. Separated by ','. Default None."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the image or video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="[image to video] The image to generate the video from.")
    parser.add_argument(
        "--first_frame",
        type=str,
        default=None,
        help="[first-last frame to video] The image (first frame) to generate the video from."
    )
    parser.add_argument(
        "--last_frame",
        type=str,
        default=None,
        help="[first-last frame to video] The image (last frame) to generate the video from."
    )
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")

    args = parser.parse_args()

    return args

def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def delete_model(model):
    import gc
    import torch

    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()



def generate_t2v(args):
    return 
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank=0)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model,
                is_vl="i2v" in args.task or "flf2v" in args.task)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                is_vl="i2v" in args.task,
                device=rank)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]


    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    logging.info(f"Input prompt: {args.prompt}")
    if args.use_prompt_extend:
        logging.info("Extending prompt ...")
        if rank == 0:
            prompt_output = prompt_expander(
                args.prompt,
                tar_lang=args.prompt_extend_target_lang,
                seed=args.base_seed)
            if prompt_output.status == False:
                logging.info(
                    f"Extending prompt failed: {prompt_output.message}")
                logging.info("Falling back to original prompt.")
                input_prompt = args.prompt
            else:
                input_prompt = prompt_output.prompt
            input_prompt = [input_prompt]
        else:
            input_prompt = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(input_prompt, src=0)
        args.prompt = input_prompt[0]
        logging.info(f"Extended prompt: {args.prompt}")

    logging.info("Creating WanT2V pipeline.")
    wan_t2v = wan.WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
    )

    logging.info(
        f"Generating {'image' if 't2i' in args.task else 'video'} ...")
    video = wan_t2v.generate(
        args.prompt,
        size=SIZE_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model)

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                     "_")[:50]
            suffix = '.png' if "t2i" in args.task else '.mp4'
            args.save_file = f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{args.ring_size}_{formatted_prompt}_{formatted_time}" + suffix

        logging.info(f"Saving generated video to {args.save_file}")
        cache_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
    logging.info("Finished.")

    delete_model(wan_t2v)


def generate_flf2v(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank=0)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model,
                is_vl="i2v" in args.task or "flf2v" in args.task)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                is_vl="i2v" in args.task,
                device=rank)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    
    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.first_frame is None or args.last_frame is None:
        args.first_frame = EXAMPLE_PROMPT[args.task]["first_frame"]
        args.last_frame = EXAMPLE_PROMPT[args.task]["last_frame"]
    logging.info(f"Input prompt: {args.prompt}")
    logging.info(f"Input first frame: {args.first_frame}")
    logging.info(f"Input last frame: {args.last_frame}")
    first_frame = Image.open(args.first_frame).convert("RGB")
    last_frame = Image.open(args.last_frame).convert("RGB")
    if args.use_prompt_extend:
        logging.info("Extending prompt ...")
        if rank == 0:
            prompt_output = prompt_expander(
                args.prompt,
                tar_lang=args.prompt_extend_target_lang,
                image=[first_frame, last_frame],
                seed=args.base_seed)
            if prompt_output.status == False:
                logging.info(
                    f"Extending prompt failed: {prompt_output.message}")
                logging.info("Falling back to original prompt.")
                input_prompt = args.prompt
            else:
                input_prompt = prompt_output.prompt
            input_prompt = [input_prompt]
        else:
            input_prompt = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(input_prompt, src=0)
        args.prompt = input_prompt[0]
        logging.info(f"Extended prompt: {args.prompt}")

    logging.info("Creating WanFLF2V pipeline.")
    wan_flf2v = wan.WanFLF2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
    )

    logging.info("Generating video ...")
    video = wan_flf2v.generate(
        args.prompt,
        first_frame,
        last_frame,
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model)

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                     "_")[:50]
            suffix = '.png' if "t2i" in args.task else '.mp4'
            args.save_file = f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{args.ring_size}_{formatted_prompt}_{formatted_time}" + suffix

        if "t2i" in args.task:
            logging.info(f"Saving generated image to {args.save_file}")
            cache_image(
                tensor=video.squeeze(1)[None],
                save_file=args.save_file,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
        else:
            logging.info(f"Saving generated video to {args.save_file}")
            cache_video(
                tensor=video[None],
                save_file=args.save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
    logging.info("Finished.")
    delete_model(wan_flf2v)



os.makedirs(SAVE_ROOT, exist_ok=True)
global args
args = _parse_args()

@register_tool("generate_t2v")
class GenerateT2VTool(BaseTool):
    description = "Generate video with provided text, and return the path of the generated video"
    parameters = [
        {
            "name": "prompt",
            "type": "string",
            "description": "input refined global prompt."
        },
    ]

    def call(self, params: dict, **kwargs):
        # qwen-agent 传进来的是 str，这里和你其它工具保持一致
        params = eval(params)
        prompt = params.get("prompt", None)

        args.task = 't2v-1.3B'
        args.size = '832*480'
        args.ckpt_dir = T2V_MODEL_CKPT
        args.offload_model = False
        args.t5_cpu = True
        args.sample_shift = 8
        args.sample_guide_scale = 6
        args.prompt = prompt
        args.save_file = SAVE_ROOT + '/global.mp4'

        _validate_args(args)

        # ===== 环境信息 =====
        rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        device = local_rank
        _init_logging(rank=0)

        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")

        if args.use_prompt_extend:
            if args.prompt_extend_method == "dashscope":
                prompt_expander = DashScopePromptExpander(
                    model_name=args.prompt_extend_model,
                    is_vl="i2v" in args.task or "flf2v" in args.task)
            elif args.prompt_extend_method == "local_qwen":
                prompt_expander = QwenPromptExpander(
                    model_name=args.prompt_extend_model,
                    is_vl="i2v" in args.task,
                    device=rank)
            else:
                raise NotImplementedError(
                    f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

        cfg = WAN_CONFIGS[args.task]
        if args.ulysses_size > 1:
            assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

        logging.info(f"Generation job args: {args}")
        logging.info(f"Generation model config: {cfg}")

        if dist.is_initialized():
            base_seed = [args.base_seed] if rank == 0 else [None]
            dist.broadcast_object_list(base_seed, src=0)
            args.base_seed = base_seed[0]


        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        logging.info(f"Input prompt: {args.prompt}")
        if args.use_prompt_extend:
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(
                    args.prompt,
                    tar_lang=args.prompt_extend_target_lang,
                    seed=args.base_seed)
                if prompt_output.status == False:
                    logging.info(
                        f"Extending prompt failed: {prompt_output.message}")
                    logging.info("Falling back to original prompt.")
                    input_prompt = args.prompt
                else:
                    input_prompt = prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info(f"Extended prompt: {args.prompt}")

        logging.info("Creating WanT2V pipeline.")
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )

        logging.info(
            f"Generating {'image' if 't2i' in args.task else 'video'} ...")
        video = wan_t2v.generate(
            args.prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)

        if rank == 0:
            if args.save_file is None:
                formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                        "_")[:50]
                suffix = '.png' if "t2i" in args.task else '.mp4'
                args.save_file = f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{args.ring_size}_{formatted_prompt}_{formatted_time}" + suffix

            logging.info(f"Saving generated video to {args.save_file}")
            cache_video(
                tensor=video[None],
                save_file=args.save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
        logging.info("Finished.")

        delete_model(wan_t2v)
        torch.cuda.empty_cache()

        return {
            "save_file": args.save_file,
        }


@register_tool("generate_flf2v")
class GenerateFLF2VTool(BaseTool):
    description = "Given the first and the last frame as well as the prompt, generate a video and return the save path"
    parameters = [
        {
            "name": "prompts",
            "type": "array",
            "items": {"type": "string"},
            "description": "A list of refined prompts, each corresponding to a video segment."
        },
        {
            "name": "first_frames",
            "type": "array",
            "items": {"type": "string"},
            "description": "A list of first frame paths, aligned with the prompts list."
        },
        {
            "name": "last_frames",
            "type": "array",
            "items": {"type": "string"},
            "description": "A list of last frame paths, aligned with the prompts list."
        },
    ]

    def call(self, params: dict, **kwargs):
        params = eval(params)
        prompt_list = params.get("prompts", None)
        first_frame_list = params.get("first_frames", None)
        last_frame_list = params.get("last_frames", None)

        args.task = 'flf2v-14B'
        args.size = '1280*720'
        args.t5_cpu = True 
        args.ckpt_dir = FLF2V_MODEL_CKPT

        _validate_args(args)
    
        rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        device = local_rank
        _init_logging(rank=0)

        args.offload_model = True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")

        cfg = WAN_CONFIGS[args.task]

        logging.info(f"Generation job args: {args}")
        logging.info(f"Generation model config: {cfg}")

        if dist.is_initialized():
            base_seed = [args.base_seed] if rank == 0 else [None]
            dist.broadcast_object_list(base_seed, src=0)
            args.base_seed = base_seed[0]

        logging.info("Creating WanFLF2V pipeline.")
        wan_flf2v = wan.WanFLF2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )

        for i in range(len(prompt_list)):
            args.prompt = prompt_list[i]
            args.first_frame = first_frame_list[i]
            args.last_frame = last_frame_list[i]
            logging.info(f"Input prompt: {args.prompt}")
            logging.info(f"Input first frame: {args.first_frame}")
            logging.info(f"Input last frame: {args.last_frame}")
            first_frame = Image.open(args.first_frame).convert("RGB")
            last_frame = Image.open(args.last_frame).convert("RGB")

            logging.info("Generating video ...")
            with torch.no_grad():
                video = wan_flf2v.generate(
                    args.prompt,
                    first_frame,
                    last_frame,
                    max_area=MAX_AREA_CONFIGS[args.size],
                    frame_num=args.frame_num,
                    shift=args.sample_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sample_steps,
                    guide_scale=args.sample_guide_scale,
                    seed=args.base_seed,
                    offload_model=args.offload_model)

            video_cpu = video.detach().cpu()
            del video
            torch.cuda.empty_cache()

            save_file = f'{SAVE_ROOT}/{i}_{i+1}.mp4'
            logging.info(f"Saving generated video to {save_file}")

            cache_video(
                tensor=video_cpu[None],
                save_file=save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1)
            )

            del video_cpu, first_frame, last_frame
            gc.collect()
            torch.cuda.empty_cache()

        logging.info("Finished.")
        delete_model(wan_flf2v)

        return {
            "save_dir": SAVE_ROOT,
        }