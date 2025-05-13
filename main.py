from modelscope import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import torch
import os
import time

def generate_video_from_image(image, prompt, num_frames=16, progress=gr.Progress()):
    try:
        progress(0, desc="加载模型中...")
        # 加载模型和分词器
        model_path = os.path.join(os.path.dirname(__file__), 'models')
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map='auto'
        )

        progress(0.3, desc="处理输入图片...")
        # 准备输入
        inputs = tokenizer.build_conversation_input_ids(
            image=image,
            text=f"请基于这张图片生成一个视频。{prompt}",
            history=[],
        )
        
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
        # 获取生成的视频路径
        video_path = outputs['video_path'][0]
        
        progress(1.0, desc="完成！")
        return video_path
        
    except Exception as e:
        progress(1.0, desc="发生错误！")
        return f"发生错误: {str(e)}"

# 创建Gradio界面
def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# 万相2.1 图生视频演示")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="filepath", label="上传图片")
                prompt_input = gr.Textbox(
                    label="提示词",
                    placeholder="请输入描述视频动作的提示词，例如：'使图片中的人物微笑'"
                )
                frames_input = gr.Slider(
                    minimum=8,
                    maximum=32,
                    value=16,
                    step=8,
                    label="视频帧数"
                )
                generate_btn = gr.Button("生成视频")
            
            with gr.Column():
                video_output = gr.Video(label="生成的视频")
        
        generate_btn.click(
            fn=generate_video_from_image,
            inputs=[image_input, prompt_input, frames_input],
            outputs=video_output,
            show_progress=True
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True, server_name="0.0.0.0")