from modelscope import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import torch
import os
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量存储加载的模型和分词器
global_model = None
global_tokenizer = None

def load_model():
    """加载模型和分词器"""
    global global_model, global_tokenizer
    try:
        if global_model is None:
            model_path = os.path.join(os.path.dirname(__file__), 'models')
            logger.info(f"开始加载模型，路径: {model_path}")
            
            global_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            global_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map='auto'
            )
            logger.info("模型加载成功")
        return global_model, global_tokenizer
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise e

def generate_video_from_image(image, prompt, num_frames=16, progress=gr.Progress()):
    try:
        if not image:
            raise ValueError("请先上传图片")
        if not prompt:
            raise ValueError("请输入提示词")
            
        logger.info(f"开始处理图片生成视频，提示词: {prompt}")
        progress(0, desc="加载模型中...")
        
        # 加载模型和分词器
        model, tokenizer = load_model()
        
        progress(0.3, desc="处理输入图片...")
        # 准备输入
        inputs = tokenizer.build_conversation_input_ids(
            image=image,
            text=f"请基于这张图片生成一个视频。{prompt}",
            history=[],
        )
        
        logger.info("开始生成视频")
        progress(0.5, desc="生成视频中...")
        # 生成视频
        outputs = model.generate(
            input_ids=inputs['input_ids'].cuda(),
            attention_mask=inputs['attention_mask'].cuda(),
            max_new_tokens=512,
            num_frames=num_frames,
            temperature=0.8,
            top_p=0.9,
        )
        
        progress(0.9, desc="处理输出视频...")
        video_path = outputs['video_path'][0]
        logger.info(f"视频生成成功，保存路径: {video_path}")
        
        progress(1.0, desc="完成！")
        return video_path
        
    except Exception as e:
        error_msg = f"发生错误: {str(e)}"
        logger.error(error_msg)
        progress(1.0, desc="发生错误！")
        return gr.Error(error_msg)

# 创建Gradio界面
def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# 万相2.1 图生视频演示")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="filepath", label="上传图片")
                prompt_input = gr.Textbox(
                    label="提示词",
                    placeholder="请输入描述视频动作的提示词，例如：'使图片中的人物微笑'",
                    value=""
                )
                frames_input = gr.Slider(
                    minimum=8,
                    maximum=32,
                    value=16,
                    step=8,
                    label="视频帧数"
                )
                generate_btn = gr.Button("生成视频", variant="primary")
                status_text = gr.Textbox(label="状态", interactive=False)
            
            with gr.Column():
                video_output = gr.Video(label="生成的视频")
        
        def on_generate_click(image, prompt, frames):
            try:
                return generate_video_from_image(image, prompt, frames)
            except Exception as e:
                return gr.Error(str(e))
        
        generate_btn.click(
            fn=on_generate_click,
            inputs=[image_input, prompt_input, frames_input],
            outputs=video_output,
            show_progress=True
        )
    
    return demo

if __name__ == "__main__":
    try:
        # 预加载模型
        logger.info("正在预加载模型...")
        load_model()
        logger.info("模型预加载完成，启动Web界面")
        
        # 启动Gradio界面
        demo = create_interface()
        demo.launch(
            share=True, 
            server_name="0.0.0.0",
            debug=True
        )
    except Exception as e:
        logger.error(f"程序启动失败: {str(e)}")