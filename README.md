# VideoAgent

## Quick Start
### Environment Setup
1. Firstly, you should meet the environment requirements of [Wan 2.1](https://github.com/Wan-Video/Wan2.1)
2. install the package for agent framework and image processing tools
```
pip install -U "qwen-agent[gui,rag,code_interpreter,mcp]"
pip install moviepy
```

### Download models
We use the Wan 2.1 T2V-1.3B and FLF2V-14B as the base model. You should download them before running.
```
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B
huggingface-cli download Wan-AI/Wan2.1-FLF2V-14B-720P --local-dir ./Wan2.1-FLF2V-14B-720P
```


### Setup the LLM API
You need to modify the api and other config in `env.py`.

### Infer
```
python main.py
```
Notice: the memory requirements are at least 80 G. 

### Others
1. You can refer to the `tools.py` and `video_gen.py` for agent defination  
2. The demo agent information flow is demonsrated in  `output.log` and correspoding videos and extracted frames are generated under the `./sample`.