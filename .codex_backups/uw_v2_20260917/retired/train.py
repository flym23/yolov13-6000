import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['WANDB_DISABLED'] = 'true'

from ultralytics import YOLO

# ==================== 加载模型 ====================
model = YOLO('/home/rom305/zzf/yolov13-305/ultralytics/cfg/models/v13/yolov13-ducra-v7.yaml')
model.load('/home/rom305/zzf/yolov13-305/yolov13n.pt')

# ==================== 开始训练 ====================
results = model.train(
    data='/home/rom305/zzf/yolov13-305/data.yaml',
    epochs=200,
    patience=40,
    batch=16,          # A100 可以先从 16 试，显存够再到 32
    workers=8,         # Linux/A100 服务器不要用 0
    amp=False,         # 服务器2 Blackwell环境统一使用FP32，避免AMP非法显存访问
    deterministic=False,  # 提速，但结果不再严格逐次复现
    plots=False,       # 减少训练过程绘图开销
    project='/home/rom305/zzf/yolov13-305/runs/train',
    name='exp'
)
