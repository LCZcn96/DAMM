import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from data_utils import get_cancer_datasets
from sklearn.model_selection import StratifiedShuffleSplit
from lifelines.utils import concordance_index
from utils import set_seed
set_seed(42)

# DeepSurv模型定义
class DeepSurv(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DeepSurv, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

# DeepSurv损失函数
class NegativeLogLikelihood(nn.Module):
    def __init__(self):
        super(NegativeLogLikelihood, self).__init__()
    
    def forward(self, risk_pred, y, e):
        hazard_ratio = torch.exp(risk_pred)
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
        uncensored_likelihood = risk_pred.T - log_risk
        censored_likelihood = uncensored_likelihood * e
        num_observed_events = torch.sum(e)
        return -torch.sum(censored_likelihood) / num_observed_events

# 训练函数
def train_deepsurv(model, train_loader, criterion, optimizer, device):
    model.train()
    for task_inputs1, task_inputs2, task_inputs3, task_inputs4, labels in train_loader:
        inputs = torch.cat((task_inputs1, task_inputs2, task_inputs3, task_inputs4), dim=1)
        inputs = inputs.to(device)
        times = labels[-3][:, 0].to(device)
        events = labels[-3][:, 1].to(device)
        optimizer.zero_grad()
        risk_pred = model(inputs)
        loss = criterion(risk_pred, times, events)
        loss.backward()
        optimizer.step()

# 评估函数
def evaluate_survival(model, data_loader, device):
    model.eval()
    all_preds, all_times, all_events = [], [], []
    with torch.no_grad():
        for task_inputs1, task_inputs2, task_inputs3, task_inputs4, labels in data_loader:
            inputs = torch.cat((task_inputs1, task_inputs2, task_inputs3, task_inputs4), dim=1)
            inputs = inputs.to(device)
            times = labels[-3][:, 0].cpu().numpy()
            events = labels[-3][:, 1].cpu().numpy()
            risk_pred = model(inputs).cpu().numpy()
            all_preds.extend(risk_pred)
            all_times.extend(times)
            all_events.extend(events)
    return concordance_index(all_times, -np.array(all_preds), all_events)

# 主函数
def main():
    # 获取数据集
    data_dict = get_cancer_datasets()
    data = data_dict["dataset"]
    
    # 数据集分层类型
    data_type = data.labels[1].cpu().numpy()
    
    # 交叉验证参数
    n_splits = 5
    batch_size = 1024
    
    # 使用StratifiedShuffleSplit进行分层交叉验证
    kf = StratifiedShuffleSplit(n_splits=n_splits, random_state=42)
    
    # 初始化结果存储
    results = {
        'DeepSurv': [],
    }
    
    for fold, (train_indices, val_indices) in enumerate(kf.split(data, data_type)):
        print(f"Fold {fold+1}/{n_splits}")
        
        # 创建训练集和验证集的采样器
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        # 创建数据加载器
        train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(data, batch_size=batch_size, sampler=val_sampler)
        
        # 训练和评估DeepSurv
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = 380501
        hidden_dim = 128
        deepsurv = DeepSurv(input_dim, hidden_dim).to(device)
        criterion = NegativeLogLikelihood()
        optimizer = torch.optim.Adam(deepsurv.parameters())
        
        for epoch in range(100): 
            train_deepsurv(deepsurv, train_loader, criterion, optimizer, device)
        
        deepsurv_c_index = evaluate_survival(deepsurv, val_loader, device)
        results['DeepSurv'].append(deepsurv_c_index)
        
        print(f"DeepSurv - C-index: {deepsurv_c_index:.4f}")
    
    # 输出平均结果
    for model in results:
        avg_c_index = np.mean(results[model])
        print(f"{model} - Avg C-index: {avg_c_index:.4f}")

if __name__ == "__main__":
    main()