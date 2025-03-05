import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# 自定义数据集类，用于处理交通数据
class TrafficDataset(Dataset):
    def __init__(self, df, seq_len):
        # 初始化数据集，接收数据框和序列长度作为参数
        self.df = df
        self.seq_len = seq_len
        # 调用 create_sequence 方法创建序列、标签和时间戳
        self.sequence, self.labels, self.timestamp = self.create_sequence(df, seq_len)

    def create_sequence(self, df, seq_len):
        # 创建时间序列数据
        # 使用 MinMaxScaler 对数据进行归一化处理
        sc = MinMaxScaler()
        # 将数据框的索引转换为 numpy 数组
        index = df.index.to_numpy()
        # 对数据框中的 'value' 列进行归一化处理，并将其转换为二维数组
        ts = sc.fit_transform(df.value.to_numpy().reshape(-1, 1))
        
        # 初始化序列、标签和时间戳列表
        sequence = []
        label = []
        timestamp = []
        # 遍历归一化后的数据，创建序列和对应的标签
        for i in range(len(ts) - seq_len):
            # 将当前序列添加到序列列表中
            sequence.append(ts[i:i+seq_len])
            # 将当前序列的下一个值作为标签添加到标签列表中
            label.append(ts[i+seq_len])
            # 将当前序列的下一个时间戳添加到时间戳列表中
            timestamp.append(index[i+seq_len])
        
        # 将列表转换为 numpy 数组并返回
        return np.array(sequence), np.array(label), np.array(timestamp)
    
    def __len__(self):
        # 返回数据集的长度，即序列的数量
        return len(self.df) - self.seq_len
    
    def __getitem__(self, idx):
        # 根据索引获取数据集中的一个样本
        # 将序列和标签转换为 torch 张量，并调整序列的维度
        return (torch.tensor(self.sequence[idx], dtype = torch.float).permute(1, 0), 
                torch.tensor(self.labels[idx], dtype = torch.float))

# 自定义数据模块类，用于管理数据集和数据加载器
class DataModule(pl.LightningDataModule):
    def __init__(self, df, seq_len):
        # 初始化数据模块，接收数据框和序列长度作为参数
        super().__init__()
        self.df = df
        self.seq_len = seq_len
    
    def setup(self, stage=None):
        # 设置数据集，在训练和预测阶段之前调用
        self.dataset = TrafficDataset(self.df, self.seq_len)
        
    def train_dataloader(self):
        # 返回训练数据加载器
        return DataLoader(self.dataset, batch_size = 32, num_workers = 10, pin_memory = True, shuffle = True)
    
    def predict_dataloader(self):
        # 返回预测数据加载器
        return DataLoader(self.dataset, batch_size = 1, num_workers = 10, pin_memory = True, shuffle = False)

# 定义 DeepAnt 模型类
class DeepAnt(nn.Module):
    def __init__(self, seq_len, p_w):
        # 初始化模型，接收序列长度和预测窗口大小作为参数
        super().__init__()
        
        # 第一个卷积块，包含卷积层、ReLU 激活函数和最大池化层
        self.convblock1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding='valid'),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # 第二个卷积块，包含卷积层、ReLU 激活函数和最大池化层
        self.convblock2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding='valid'),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # 扁平化层，将卷积层的输出展平为一维向量
        self.flatten = nn.Flatten()
        
        # 全连接块，包含线性层、ReLU 激活函数和 Dropout 层
        self.denseblock = nn.Sequential(
            nn.Linear(32, 40),
            #nn.Linear(96, 40), # for SEQL_LEN = 20
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
        )
        # 输出层，将全连接块的输出映射到预测窗口大小
        self.out = nn.Linear(40, p_w)
        
    def forward(self, x):
        # 前向传播方法，定义模型的计算流程
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.flatten(x)
        x = self.denseblock(x)
        x = self.out(x)
        return x

# 自定义异常检测模型类，继承自 PyTorch Lightning 的 LightningModule
class AnomalyDetector(pl.LightningModule):
    def __init__(self, model):
        # 初始化异常检测模型，接收一个模型作为参数
        super().__init__()
        self.model = model
        # 使用 L1 损失函数作为训练的损失函数
        self.criterion = nn.L1Loss()
    
    def forward(self, x):
        # 前向传播方法，调用传入的模型进行计算
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        # 训练步骤，定义每个训练批次的计算流程
        # 从批次中获取输入数据和标签
        x, y = batch
        # 调用模型进行预测
        y_pred = self(x)
        # 计算预测值和标签之间的损失
        loss = self.criterion(y_pred, y)
        # 记录训练损失，用于日志记录和可视化
        self.log('train_loss', loss, prog_bar=True, logger = True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        # 预测步骤，定义每个预测批次的计算流程
        # 从批次中获取输入数据和标签
        x, y = batch
        # 调用模型进行预测
        y_pred = self(x)
        # 计算预测值和标签之间的范数，用于异常检测
        return y_pred, torch.linalg.norm(y_pred-y)
    
    def configure_optimizers(self):
        # 配置优化器，使用 Adam 优化器，学习率为 1e-5
        return torch.optim.Adam(self.parameters(), lr = 1e-5)
    