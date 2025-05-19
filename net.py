# net.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.tensorboard import SummaryWriter

class PG_Net(nn.Module):
    """
    带有 CNN 层#(和残差块)的策略网络。
    接收堆叠帧作为输入，并输出每个动作的概率。
    """
    def __init__(self, input_shape, num_actions):
        """
        Args:
            input_shape (tuple): 输入状态的形状 (例如, (帧数, 高度, 宽度) -> (4, 256, 256))。
            num_actions (int): 可能的动作数量。
        """
        super(PG_Net, self).__init__()
        num_frames = input_shape[0] # 4
        height = input_shape[1]     # 256
        width = input_shape[2]      # 256

        # --- 卷积层 ---
        # 初始卷积层，处理堆叠的帧。大核、大步长、升维，快速提取大尺度特征，降低分辨率，升维增加特征数量
        # 输入: (批次大小, 帧数, 高度, 宽度) -> (B, 4, 256, 256)
        self.conv1 = nn.Conv2d(in_channels=num_frames, out_channels=32, kernel_size=8, stride=4,padding=2)
        # 输出形状: 256/4 = 64 -> (B, 32, 64, 64)
        self.relu1 = nn.ReLU(inplace=True)
        # 输出形状: 64/2=32 -> (B, 64, 32, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        

        # 第二个卷积层，下采样和增加特征数量
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2,padding=1)
        # 输出形状: 32/2=16 -> (B, 64, 16, 16)
        self.relu2 = nn.ReLU(inplace=True)
        # 输出形状: 16/2=8 -> (B, 64, 8, 8)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第三个卷积层
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,padding=1)
         # 输出形状: 8/1=8 -> (B, 64, 8, 8)
        self.relu3 = nn.ReLU(inplace=True)

        # --- 展平层和全连接层 ---
        self.flatten = nn.Flatten()

        # 根据 input_shape 动态计算展平后的大小
        with torch.no_grad(): # 不需要计算梯度
            dummy_input = torch.zeros(1, *input_shape) # (1, 4, 256, 256)
            conv_out_size = self._get_conv_output_size(dummy_input)
        print(f"计算得到的卷积层输出大小: {conv_out_size}")

        # 全连接层
        self.fc1 = nn.Linear(conv_out_size, 1024)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024,256)
        self.relu5 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(256, num_actions) # 输出层 (每个动作分数)

        # 初始化权重
        self._initialize_weights()


    def _get_conv_output_size(self, x):
        """ 辅助函数，计算通过卷积层后的大小 """
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.relu3(self.conv3(x))
        x = self.flatten(x)
        return x.shape[1] # 返回展平后特征的大小

    def _initialize_weights(self):
        """ 使用 He 初始化或 Orthogonal 初始化来初始化卷积层和线性层的权重 """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # Kaiming 初始化
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2)) # Orthogonal 初始化通常效果也很好
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0) # 偏置初始化为 0
            elif isinstance(m, nn.Linear):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # Kaiming 初始化
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2)) # Orthogonal 初始化
                nn.init.constant_(m.bias, 0) # 偏置初始化为 0

    def forward(self, x):
        """
        通过网络的前向传播。

        Args:
            x (torch.Tensor): 输入张量 (一批堆叠的帧)。
                             需要是 float 类型并且已经归一化 (例如, 归一化到 [0, 1])。
                             形状: (批次大小, 帧数, 高度, 宽度)

        Returns:
            torch.Tensor: 每个动作的分数。形状: (批次大小, 动作数量)
        """

        # 将数字类型转换（uint8->float32）放到agent.py中进行

        # 卷积层
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.relu3(self.conv3(x))

        # 展平与全连接层
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        x = self.relu5(x)
        logits = self.fc3(x) 
        #print(logits)

        return logits

# 实例化模型
input_shape = (4, 256, 256)
num_actions = 4
model = PG_Net(input_shape, num_actions)

# # 创建一个 TensorBoard writer
# # 'runs/qnet_resnet_graph' 是日志文件存放目录，可以自定义
# writer = SummaryWriter('runs/qnet_resnet_graph')

# # 创建一个符合输入形状的虚拟输入张量 (batch_size=1)
# # 注意：输入需要是 float 类型
# dummy_input = torch.randn(1, *input_shape)

# # 将模型图写入 TensorBoard
# writer.add_graph(model, dummy_input)
# writer.close()

# print("模型图已写入 TensorBoard 日志文件。")
# print("请在终端运行以下命令启动 TensorBoard:")
# print(f"tensorboard --logdir=runs/qnet_resnet_graph")
# print("然后在浏览器中打开显示的 URL (通常是 http://localhost:6006/)")
