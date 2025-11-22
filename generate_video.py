# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_image, cache_video, str2bool
from generate import EXAMPLE_PROMPT
import cv2
import os
import glob

PROMPT = 'a man walk along the street.'
SAVE_ROOT = '/m2v_intern/yangzhenhao/code/video_agent/Wan2.1/demo'
TARGET_SIZE = '1280*720'


def extract_frames_by_index(video_path, frame_indices, out_dir="frames_by_index"):
    print("----------Calling extract_frames_by_index functon----------")
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    for idx in frame_indices:
        # 设置要读取的帧号
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        success, frame = cap.read()
        if not success:
            print(f"⚠️ 无法读取第 {idx} 帧（可能超过视频总帧数）")
            continue

        out_path = os.path.join(out_dir, f"frame_{idx:06d}.png")
        cv2.imwrite(out_path, frame)
        print(f"已保存：{out_path}")

    cap.release()

def concat_videos_timewise(input_dir):
    """
    将 input_dir 中所有视频按照文件名排序后
    在时间维度上顺序拼接成一个输出视频。
    """
    print("----------Calling concat_videos_timewise functon----------")
    output_path = os.path.join(input_dir, 'final_output.mp4')
    # 搜集视频
    video_files = []
    for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))

    if not video_files:
        raise RuntimeError("路径内未找到视频文件")

    # 根据文件名排序
    video_files = sorted(video_files)
    video_files = [x for x in video_files if not x.startswith('global')]

    # 用第一个视频读取关键参数
    cap = cv2.VideoCapture(video_files[0])
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"[INFO] 使用分辨率: {width}x{height}, fps: {fps}")

    # 创建输出视频
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 遍历每个视频，按顺序写入
    for path in video_files:
        print(f"[INFO] 拼接: {path}")
        cap = cv2.VideoCapture(path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 可选：检查 resolution 不一致 -> 自动 resize
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            writer.write(frame)

        cap.release()

    writer.release()
    print(f"[OK] 输出完成: {output_path}")


def resize_images(input_dir, target_size):
    """
    input_dir: 输入图片路径
    target_size: (width, height) 指定大小
    """
    # 支持常见图片格式
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))

    print(f"[INFO] 发现 {len(image_files)} 张图片")

    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] 无法读取: {img_path}")
            continue

        # 变形 (resize)
        resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

        # 输出路径
        out_path = os.path.join(input_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, resized)
        print(f"[OK] 保存: {out_path}")


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


if __name__ == "__main__":
    args = _parse_args()
    os.makedirs(SAVE_ROOT, exist_ok=True)

    # # TODO: add logic to refine the prompt for global video generation

    # # generate global video
    # args.task = 't2v-1.3B'
    # args.sample_guide_scale = 6
    # args.sample_shift = 8
    # args.t5_cpu = True
    # args.offload_model = True
    # args.ckpt_dir = './Wan2.1-T2V-1.3B'
    # args.size = '832*480'
    # args.prompt = PROMPT
    # args.save_file = os.path.join(SAVE_ROOT, 'global_video.mp4')
    # _validate_args(args)

    # generate_t2v(args)
    
    # # distill fisrt and last frame
    # frame_indices = [0, 20, 40]
    # extract_frames_by_index(args.save_file, frame_indices=frame_indices, out_dir=SAVE_ROOT)


    # # resize 1.3b output to target size    
    # resize_images(SAVE_ROOT, target_size=tuple(map(int, TARGET_SIZE.split('*'))))

    # generate mutishot video
    args.task = 'flf2v-14B'
    args.size = TARGET_SIZE
    args.sample_shift = 16
    args.sample_guide_scale = 5.0
    args.prompt = PROMPT
    args.ckpt_dir = './Wan2.1-FLF2V-14B-720P'

    _validate_args(args)
    file_list = sorted(os.listdir(SAVE_ROOT))
    file_list = [x for x in file_list if not x.startswith('global')]
    files_pair = [(file_list[i], file_list[i+1]) for i in range(len(file_list)-1)]
    print(files_pair)
    for first_frame, last_frame in files_pair:
        args.save_file = os.path.join(SAVE_ROOT, f"{first_frame.split('.')[0]}_{first_frame.split('.')[1]}.mp4")
        args.first_frame = os.path.join(SAVE_ROOT, first_frame)
        args.last_frame = os.path.join(SAVE_ROOT, last_frame)
        generate_flf2v(args)
    
    # concat the multishot video to get the final output
    concat_videos_timewise(SAVE_ROOT)