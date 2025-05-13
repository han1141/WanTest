import os
import gc
import torch
from PIL import Image
import gradio as gr
import time
from skyreels_v2_infer.modules import download_model
from skyreels_v2_infer.pipelines import Image2VideoPipeline, resizecrop
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_available_tasks():
    """检查 ModelScope 支持的任务类型"""
    try:
        # 正确方式获取支持的任务
        available_tasks = [task for task in dir(Tasks) if not task.startswith('_')]
        logger.info("支持的任务类型：")
        for task in available_tasks:
            logger.info(f"- {task}")
        return available_tasks
    except Exception as e:
        logger.error(f"获取任务类型失败: {str(e)}")
        return []

def check_model_config(model_dir):
    """检查模型配置"""
    config_path = os.path.join(model_dir, 'configuration.json')
    if not os.path.exists(config_path):
        raise ValueError(f"模型配置文件不存在: {config_path}")
    
    try:
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            if 'pipeline' not in config:
                logger.warning("配置文件中缺少 pipeline 字段")
    except Exception as e:
        logger.error(f"读取配置文件失败: {str(e)}")

def verify_model_files(model_dir):
    """验证模型文件的完整性"""
    required_files = ['configuration.json', 'config.json']
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        raise ValueError(f"模型目录缺少以下文件: {', '.join(missing_files)}")
    
    return True

def generate_video_from_image(image_path, prompt, num_frames=16, progress=gr.Progress()):
    try:
        progress(0, desc="准备模型...")
        model_id = "Skywork/SkyReels-V2-I2V-1.3B-540P"
        model_dir = download_model(model_id)
        height, width = 544, 960  # 540P分辨率
        negative_prompt = (
            "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
            "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
            "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
            "messy background, three legs, many people in the background, walking backwards"
        )

        # 处理图片
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size
        if image_height > image_width:
            height, width = width, height
        image = resizecrop(image, height, width)

        progress(0.2, desc="初始化推理管线...")
        pipe = Image2VideoPipeline(
            model_path=model_dir,
            dit_path=model_dir,
            use_usp=False,
            offload=False
        )

        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_frames": num_frames,
            "num_inference_steps": 30,
            "guidance_scale": 6.0,
            "shift": 8.0,
            "generator": generator,
            "height": height,
            "width": width,
            "image": image
        }

        progress(0.5, desc="生成视频帧...")
        with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
            video_frames = pipe(**kwargs)[0]

        # 保存视频
        save_dir = os.path.join("result", "webui")
        os.makedirs(save_dir, exist_ok=True)
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        video_out_file = f"{prompt[:100].replace('/','')}_webui_{current_time}.mp4"
        output_path = os.path.join(save_dir, video_out_file)

        import imageio
        imageio.mimwrite(output_path, video_frames, fps=24, quality=8, output_params=["-loglevel", "error"])

        gc.collect()
        torch.cuda.empty_cache()
        progress(1.0, desc="完成！")
        return output_path
    except Exception as e:
        progress(1.0, desc="发生错误！")
        return None

def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Skywork I2V 图生视频演示")
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="filepath", label="上传图片", height=320, width=320)
                prompt_input = gr.Textbox(label="提示词")
                frames_input = gr.Slider(8, 32, value=16, step=8, label="视频帧数")
                generate_btn = gr.Button("生成视频")
                status_output = gr.Textbox(label="状态", interactive=False)
            with gr.Column():
                video_output = gr.Video(label="生成的视频")

        def on_generate(image, prompt, frames):
            video_path = generate_video_from_image(image, prompt, frames)
            if video_path:
                return video_path, "生成成功！"
            return None, "生成失败！"

        generate_btn.click(
            fn=on_generate,
            inputs=[image_input, prompt_input, frames_input],
            outputs=[video_output, status_output],
            show_progress=True
        )
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", debug=True, show_error=True)