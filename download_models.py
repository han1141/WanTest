from modelscope import snapshot_download
import os

def download_wanx_model():
  # 设置模型ID
  model_id = 'Xorbits/WanXiang-2.1'
  
  # 设置下载目录
  download_path = os.path.join(os.path.dirname(__file__), 'models')
  
  try:
    # 确保下载目录存在
    os.makedirs(download_path, exist_ok=True)
    
    # 下载模型到指定目录
    model_dir = snapshot_download(
        model_id,
        cache_dir=download_path,
        revision='master'
    )
    print(f"模型下载成功！")
    print(f"模型保存位置: {model_dir}")
      
  except Exception as e:
    print(f"下载过程中发生错误: {str(e)}")

if __name__ == "__main__":
    download_wanx_model()