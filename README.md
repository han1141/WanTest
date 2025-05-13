# 万相2.1 图生视频演示项目

## 环境要求
- Python 3.10
- CUDA 11.8 及以上（如果使用GPU）
- 至少16GB显存（推荐）

## 安装步骤

1. 创建并激活Python虚拟环境：
```bash
python3.10 -m venv venv
source venv/bin/activate  # MacOS/Linux
```

2. 安装依赖：
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. 下载模型：
```bash
python download_models.py
```

4. 运行演示：
```bash
python main.py
```

## 使用说明

1. 运行后打开浏览器访问显示的URL（默认为 http://127.0.0.1:7860）
2. 上传一张图片
3. 输入描述期望生成视频效果的提示词
4. 调整视频帧数（8-32帧）
5. 点击"生成视频"按钮

## 注意事项

- 首次下载模型大约需要13GB存储空间
- 生成视频时需要较大显存，建议使用配置较高的GPU
- 如遇到CUDA相关错误，请确保CUDA版本与PyTorch版本匹配