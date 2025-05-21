# config.py

import torch


# 参数配置
CONFIG = {
    "env_id": "LunarLander-v3",
    "seed": 11, # 用于复现性
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "total_episodes": 200000, # 总训练步数
    #"learning_starts": 10000,    # 多少步之后开始学习 (填充缓冲区)
    #"buffer_capacity": 100000,   # Replay Buffer 容量
    #"batch_size": 16,           # 学习时的批大小
    "learning_rate": 1e-5,      # 学习率
    "gamma": 0.99,              # 折扣因子
    #"epsilon_start": 0.25,       # Epsilon 初始值
    #"epsilon_end": 0.02,        # Epsilon 最终值
    #"epsilon_decay_steps": 7500000, # Epsilon 从 start 线性衰减到 end 的步数
    #"target_update_frequency": 1000, # 目标网络更新频率 (按步数)
    "save_frequency": 10000,     # 每多少步保存一次模型
    "model_save_dir": "PG\model",  # 模型保存路径
    "checkpoint_path": "PG\models\PG_lunarlander_step_100000.pth",     # "PG\models/PG_lunarlander_step_xxxxx.pth" # 可选：加载检查点继续训练
    # 环境相关
    "num_frames": 4,             # 帧堆叠数量
    "img_size": (256, 256)         # 图像预处理尺寸
}