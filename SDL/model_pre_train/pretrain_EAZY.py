# pretrain.py
import math, random, json, os, warnings, itertools
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import ast   
import glob   
warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

# 日志配置：写入文件并同时输出到控制台
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "pretrain.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)


def form_to_int(raw: str) -> int:
    """
    把 'A-', 'HA' 及其大小写变体 → 0 或 1
    0 = 去质子化态 (A-)
    1 = 质子化态   (HA)
    """
    val = str(raw).strip().upper()
    if val == "A-":
        return 0
    elif val == "HA":
        return 1
    else:
        raise ValueError(f"Unexpected Initial_Form value: {raw}")
# ---------- 1. 数据读取与样本扁平化 ----------
CSV_PATH = "titration_dataset.csv"          # 原始表格
N_MAX_COMPONENTS =  4                     # 一条记录里最多物种数（可动态）
EMBED_DIM = 256                             # 体系编码器输出维度
BATCH_SIZE = 1024
EPOCHS = 30
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# logging.info(f"Using device: {DEVICE}")

CSV_DIR = r"E:\mypython\SDL\train_csv"                      # 放所有 CSV 的文件夹
csv_files = glob.glob(os.path.join(CSV_DIR, "*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV found in {CSV_DIR}")

all_dfs = [pd.read_csv(f) for f in csv_files]
df = pd.concat(all_dfs, ignore_index=True)
# logging.info(f"Total raw rows across all CSV: {len(df)}")

# ---------- 2. 构造曲线 id ----------
# 如果 CSV 里本来就有 curve_id 这一列，直接用；否则用组分信息拼一个
if "curve_id" not in df.columns:
    id_cols = [c for c in df.columns if c.startswith("Component_") and ("Name" in c or "Conc" in c)]
    df["curve_id"] = df[id_cols].astype(str).agg("|".join, axis=1)

# ---------- 3. 按 curve_id 聚合并拆成 2-D 样本 ----------
records = []
max_components_seen = 0

for cid, grp in df.groupby("curve_id"):
    # 排序保证体积递增
    grp = grp.sort_values("Volume_mL")
    vols = grp["Volume_mL"].astype(float).tolist()
    phs  = grp["pH"].astype(float).tolist()

    row0 = grp.iloc[0]

    # 动态读取所有存在的组分
    comps = []
    i = 1
    while True:
        name_key = f"Component_{i}_Name"
        if name_key not in df.columns:
            break
        name = row0.get(name_key, None)
        if pd.isna(name):
            break
        comps.append({
            "name": name,
            "pKa":  float(row0[f"Component_{i}_pKa"]),
            "conc": float(row0[f"Component_{i}_Total_Conc_M"]),
            "form": 0 if str(row0[f"Component_{i}_Initial_Form"]).strip().upper() == "A-" else 1,
            "charge": int(row0[f"Component_{i}_Counter_Ion_Charge"])
        })
        i += 1
    max_components_seen = max(max_components_seen, len(comps))

    for v, ph in zip(vols, phs):
        records.append({"comps": comps, "global": np.zeros(3), "v": v, "ph": ph})

# logging.info(f"Flattened samples: {len(records)}")
# logging.info(f"Max components per curve: {max_components_seen}")

# ---------- 2. 数据集类 ----------
class TitrationDataset(Dataset):
    def __init__(self, recs, scaler=None):
        self.recs = recs
        # 对全局标量做标准化
        globals = np.stack([r["global"] for r in recs])
        if scaler is None:
            self.scaler = StandardScaler().fit(globals)
        else:
            self.scaler = scaler
        for r in recs:
            r["global"] = self.scaler.transform(r["global"].reshape(1,-1))[0]

    def __len__(self):
        return len(self.recs)

    def augment(self, comp):
        # 轻微扰动，模拟实验噪声
        comp = comp.copy()
        comp["pKa"]   += np.random.normal(0, 0.05)
        comp["conc"]  *= np.random.normal(1, 0.02)
        return comp

    def __getitem__(self, idx):
        r = self.recs[idx]

        # 解析组分
        comps = [self.augment(c) if random.random() < 0.5 else c for c in r["comps"]]
        padded = comps + [{"pKa": 0, "conc": 0, "form": 0, "charge": 0}] * (N_MAX_COMPONENTS - len(comps))
        x_species = torch.tensor([[c["pKa"], c["conc"], c["form"], c["charge"]] for c in padded], dtype=torch.float32)

        # 去掉 Ionic_Strength & Buffer_Capacity，仅保留占位
        x_global = torch.zeros(3, dtype=torch.float32)

        v = torch.tensor([r["v"]], dtype=torch.float32)
        ph = torch.tensor([r["ph"]], dtype=torch.float32)

        return {
            "x_species": x_species,
            "x_global":  x_global,
            "v":         v,
            "ph":        ph
        }

# 划分
train_recs, val_recs = train_test_split(records, test_size=0.1, random_state=42)
train_ds = TitrationDataset(train_recs)
val_ds   = TitrationDataset(val_recs, scaler=train_ds.scaler)

# ---------- 3. 模型 ----------
class DeepSetEncoder(nn.Module):
    """
    输入: [B, N_MAX, 4] 物种特征
    输出: [B, EMBED_DIM] 体系向量
    """
    def __init__(self, d_in=4, d_hidden=256, d_out=EMBED_DIM):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out)
        )
        self.rho = nn.Sequential(
            nn.Linear(d_out, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out)
        )

    def forward(self, x):
        # x: [B, N, 4]
        h = self.phi(x)            # [B,N,EMBED]
        h = h.sum(dim=1)           # 对称池化 -> [B,EMBED]
        out = self.rho(h)          # [B,EMBED]
        return out

class VolumeEncoder(nn.Module):
    """正弦位置编码 + MLP"""
    def __init__(self, d_out=64, max_vol=50):
        super().__init__()
        self.max_vol = max_vol
        self.mlp = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, d_out)
        )
    def forward(self, v):
        # v: [B,1]
        v = v.squeeze(-1)                      # [B]
        pe = torch.zeros(v.size(0), 32, device=v.device)
        pos = v.unsqueeze(1)                   # [B,1]
        div = torch.exp(torch.arange(0, 32, 2, dtype=torch.float32, device=v.device) * -(math.log(10000.0) / 32))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return self.mlp(pe)                    # [B, d_out]

class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM + 64, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )
    def forward(self, z_system, z_vol):      
        z = torch.cat([z_system, z_vol], dim=-1)
        return self.net(z).squeeze(-1)

class TitrationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DeepSetEncoder()
        self.vol_enc = VolumeEncoder()
        self.fusion = FusionNet()

    def forward(self, x_species, x_global, v):   # 签名保持，但 x_global 不 feed
        z_sys = self.encoder(x_species)
        z_vol = self.vol_enc(v)
        return self.fusion(z_sys, z_vol)

# ---------- 4. 训练 ----------
def run_epoch(loader, training):
    total, mse = 0, 0.0
    if training:
        model.train()
    else:
        model.eval()
    for batch in tqdm(loader, desc="train" if training else "val"):
        x_sp = batch["x_species"].to(DEVICE)
        x_gl = batch["x_global"].to(DEVICE)
        v    = batch["v"].to(DEVICE)
        ph_t = batch["ph"].to(DEVICE)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            ph_p = model(x_sp, x_gl, v)
            loss = loss_fn(ph_p, ph_t)
            if training:
                loss.backward()
                optimizer.step()
        total += ph_t.size(0)
        mse   += loss.item() * ph_t.size(0)
    return math.sqrt(mse/total)
if __name__ == "__main__":

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = TitrationModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = nn.MSELoss()

    best_val = 1e9
    os.makedirs("ckpt", exist_ok=True)
    for epoch in range(1, EPOCHS+1):
        train_rmse = run_epoch(train_loader, True)
        val_rmse   = run_epoch(val_loader,   False)
        scheduler.step()
        logging.info(f"Epoch {epoch:02d} | train RMSE: {train_rmse:.3f} | val RMSE: {val_rmse:.3f}")
        # 替换原来的保存代码
        if val_rmse < best_val:
            best_val = val_rmse
            torch.save({
                "encoder": model.encoder.state_dict(),
                "fusion": model.fusion.state_dict(),
                "vol_enc": model.vol_enc.state_dict(),
                "scaler": train_ds.scaler  # 添加scaler
            }, "ckpt/pretrain_best.pt")
            logging.info(f"New best val RMSE: {best_val:.3f} -> checkpoint saved to ckpt/pretrain_best.pt")