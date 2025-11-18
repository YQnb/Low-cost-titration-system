# baseline_nn_simple.py - 简化版，只使用pH和体积信息
import math, random, os, warnings
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import glob
warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)

# ---------- 日志 ----------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "baseline_nn_simple.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()]
)

# ---------- 归一化层 ----------
class NormalizationLayer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('feature_mean', mean.clone().detach())
        self.register_buffer('feature_std', std.clone().detach())
        # 避免除零
        self.feature_std = torch.where(self.feature_std == 0, 
                                     torch.ones_like(self.feature_std), 
                                     self.feature_std)

    def forward(self, x):
        return (x - self.feature_mean) / self.feature_std

# ---------- 简化模型（只使用体积信息） ----------
class SimpleBaselineNN(nn.Module):
    def __init__(self, input_dim=1, feature_mean=None, feature_std=None):
        super().__init__()
        
        # 归一化层
        if feature_mean is not None and feature_std is not None:
            self.normalize = NormalizationLayer(feature_mean, feature_std)
        else:
            self.normalize = None

        # 简化网络结构，只预测pH
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        if self.normalize is not None:
            x = self.normalize(x)
        return self.net(x).squeeze(-1)

# ---------- 简化数据集（只使用体积） ----------
class SimpleBaselineDataset(Dataset):
    def __init__(self, csv_files, scaler=None):
        self.records = []
        
        print("🔧 加载简化数据集...")
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                filename = os.path.basename(csv_file)
                
                for _, row in df.iterrows():
                    # 只使用体积作为特征
                    volume = row["Volume_mL"]
                    ph = row["pH"]
                    
                    record = {
                        "volume": volume,
                        "ph": ph,
                        "file": filename
                    }
                    self.records.append(record)
                
                print(f"📊 加载: {filename} - {len(df)} 个数据点")
                
            except Exception as e:
                print(f"❌ 加载 {csv_file} 失败: {e}")
        
        # 准备特征
        features = np.array([[r["volume"]] for r in self.records], dtype=np.float32)
        targets = np.array([r["ph"] for r in self.records], dtype=np.float32)
        
        # 归一化
        if scaler is None:
            self.scaler = StandardScaler().fit(features)
        else:
            self.scaler = scaler
            
        self.features = features  # 原始特征
        self.targets = targets

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "ph": torch.tensor([self.targets[idx]], dtype=torch.float32),
            "file": self.records[idx]["file"]
        }

# ---------- 训练一个 epoch ----------
def run_epoch(loader, model, optimizer, loss_fn, training=True, device="cpu"):
    total, mse = 0, 0.0
    model.train() if training else model.eval()
    for batch in tqdm(loader, desc="train" if training else "val"):
        features = batch["features"].to(device)
        ph_true = batch["ph"].to(device)
        
        if training:
            optimizer.zero_grad()
            
        with torch.set_grad_enabled(training):
            ph_pred = model(features)
            loss = loss_fn(ph_pred, ph_true.squeeze())
            
            if training:
                loss.backward()
                optimizer.step()
                
        total += ph_true.size(0)
        mse += loss.item() * ph_true.size(0)
        
    return math.sqrt(mse / total)

# ---------- 主入口 ----------
if __name__ == "__main__":
    # 超参数
    BATCH_SIZE = 512
    EPOCHS = 50
    LR = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("ckpt", exist_ok=True)

    # 设置随机种子
    def set_seed(s=42):
        random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    set_seed()

    # 1. 查找CSV文件
    CSV_DIR = r"E:\mypython\SDL\train_csv"  # 使用真实数据
    csv_files = glob.glob(os.path.join(CSV_DIR, "*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {CSV_DIR}")
    
    print(f"找到 {len(csv_files)} 个CSV文件")

    # 2. 创建数据集
    all_ds = SimpleBaselineDataset(csv_files, scaler=None)
    
    # 按样本比例划分训练验证集
    all_records = all_ds.records
    all_features = all_ds.features
    all_targets = all_ds.targets

    # 随机打乱并划分
    indices = np.random.permutation(len(all_records))
    split_idx = int(0.8 * len(all_records))

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    # 创建训练集
    train_ds = SimpleBaselineDataset([], scaler=all_ds.scaler)
    train_ds.records = [all_records[i] for i in train_indices]
    train_ds.features = all_features[train_indices]
    train_ds.targets = all_targets[train_indices]

    # 创建验证集
    val_ds = SimpleBaselineDataset([], scaler=all_ds.scaler)
    val_ds.records = [all_records[i] for i in val_indices]
    val_ds.features = all_features[val_indices]
    val_ds.targets = all_targets[val_indices]

    print(f"训练集: {len(train_ds)} 样本, 验证集: {len(val_ds)} 样本")

    # 数据加载器
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 4. 模型
    train_mean = torch.from_numpy(all_ds.scaler.mean_).float().to(DEVICE)
    train_std = torch.from_numpy(all_ds.scaler.scale_).float().to(DEVICE)
    
    model = SimpleBaselineNN(input_dim=1, feature_mean=train_mean, feature_std=train_std).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = nn.MSELoss()

    # 5. 保存归一化参数
    np.save("ckpt/baseline_simple_mean.npy", all_ds.scaler.mean_)
    np.save("ckpt/baseline_simple_std.npy", all_ds.scaler.scale_)
    logging.info(f"简化模型归一化参数已保存")

    # 6. 训练循环
    best_val_rmse = float('inf')
    for epoch in range(1, EPOCHS + 1):
        train_rmse = run_epoch(train_loader, model, optimizer, loss_fn, True, DEVICE)
        val_rmse = run_epoch(val_loader, model, optimizer, loss_fn, False, DEVICE)
        scheduler.step()
        
        logging.info(f"Epoch {epoch:02d} | Train RMSE {train_rmse:.4f} | Val RMSE {val_rmse:.4f}")
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler_mean': all_ds.scaler.mean_,
                'scaler_std': all_ds.scaler.scale_,
                'input_dim': 1,
                'val_rmse': best_val_rmse,
                'epoch': epoch
            }, "ckpt/baseline_simple_best.pt")
            logging.info(f"*** 新最佳 val RMSE: {best_val_rmse:.4f} ***")

    logging.info(f"训练完成！最佳验证RMSE: {best_val_rmse:.4f}")