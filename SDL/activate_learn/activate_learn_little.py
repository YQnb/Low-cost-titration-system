# activate_learn_little.py
import torch
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scipy.stats import norm
import numpy as np
import time
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d

from torch.utils.data import DataLoader
from model_pre_train.pretrain_EAZY import TitrationModel, TitrationDataset
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from activate_learn.activate_logger import ActiveLearningLogger

# 线程安全绘图
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class ActiveTitrationLearner:
    def __init__(self, arduino_controller=None, target_ph=7.0,
                 simulate=False,
                 csv_path=r"E:\mypython\SDL\activate_learn\acetate_titration.csv"):

        self.arduino = arduino_controller
        self.logger = ActiveLearningLogger()
        self.target_ph = target_ph
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 超参数
        self.budget_real = 10
        self.lr_fine = 1e-3
        self.epochs_fine = 10
        self.min_step = 0.1
        self.max_step = 10.0
        self.hit_cnt_target = 2

        # 模型
        self.model, self.scaler = self._load_pretrained()
        self.labeled_data = []
        self.current_volume = 0.0
        self.pump0_volume = 0
        self.pump1_volume = 0

        # GP
        self.gp = GaussianProcessRegressor(
            kernel=C(1.0) * RBF(length_scale=1.0),
            n_restarts_optimizer=0)

        # 漂移检测
        self.drift_threshold = 0.2
        self.drift_retry = 3
        self.drift_wait = 5.0

        # 模拟
        self.simulate = simulate
        self.csv_path = Path(csv_path)
        if self.simulate:
            self._load_csv_interpolator()

    # ---------- CSV 插值 ----------
    def _load_csv_interpolator(self):
        df = pd.read_csv(self.csv_path)
        self._csv_v = df['Cumulative_V_mL'].astype(float).values
        self._csv_ph = df['pH'].astype(float).values
        self._csv_ph_interp = interp1d(
            self._csv_v, self._csv_ph,
            kind='linear', bounds_error=False,
            fill_value=(self._csv_ph[0], self._csv_ph[-1]))

    # ---------- 预训练权重 ----------
    def _load_pretrained(self):
        ckpt = torch.load("ckpt/pretrain_best.pt", map_location=self.device)
        model = TitrationModel().to(self.device)
        model.encoder.load_state_dict(ckpt["encoder"])
        model.vol_enc.load_state_dict(ckpt["vol_enc"])
        return model, ckpt["scaler"]

    def _load_pretrained_weights(self):
        ckpt = torch.load("ckpt/pretrain_best.pt", map_location=self.device)
        self.model.encoder.load_state_dict(ckpt["encoder"])
        self.model.vol_enc.load_state_dict(ckpt["vol_enc"])
        self.model.fusion.load_state_dict(ckpt["fusion"])

    # ---------- 稳定读数 ----------
    def _read_stable_ph(self, timeout=90):
        start = time.time()
        prev = None
        while time.time() - start < timeout:
            if self.simulate:
                net_volume = self.pump1_volume - self.pump0_volume
                cur = float(self._csv_ph_interp(net_volume))
            else:
                cur = self.arduino.read_ph()
            if cur is None:
                raise RuntimeError("pH 传感器无数据")
            if prev is None or abs(cur - prev) < self.drift_threshold:
                return cur
            prev = cur
            time.sleep(self.drift_wait)
        raise RuntimeError("90 s 内未获得稳定 pH")

    # ---------- 测量 ----------
    def _measure_ph(self, volume_added=0.0, mix_time=25.0, direction=None):
        # 先读取当前稳定pH值
        current_ph = self._read_stable_ph()
        
        # 记录当前状态（添加体积前的状态）
        action, signed_vol = "hold", 0.0
        current_pump0 = self.pump0_volume
        current_pump1 = self.pump1_volume
        current_total_vol = self.current_volume
        
        if volume_added > 0:
            # 确定滴定方向
            if direction is None:
                direction = "acid" if current_ph > self.target_ph else "base"
            
            if direction == "base":
                self.pump1_volume += volume_added
                action, signed_vol = "base", volume_added
                if not self.simulate:
                    self.arduino.pump0_flow(volume_added)
            else:  # direction == "acid"
                self.pump0_volume += volume_added
                action, signed_vol = "acid", -volume_added
                if not self.simulate:
                    self.arduino.pump1_flow(volume_added)
            
            # 更新总体积
            self.current_volume += volume_added
            
            # 混合过程
            if not self.simulate:
                self.arduino.FAN_ON()
                time.sleep(mix_time)
                self.arduino.FAN_OFF()
            
            # 读取添加后的稳定pH值
            new_ph = self._read_stable_ph()
            
            # if self.simulate:
            #     # 模拟模式：使用插值函数获取新pH值
            #     # time.sleep(0.05)
        else:
            new_ph = current_ph  # 无添加时，pH不变

        self.logger.log_step(
            cur_ph=current_ph,           # 添加前的pH
            target_ph=self.target_ph,
            pump0_ml=current_pump0,      # 添加前的泵0体积
            pump1_ml=current_pump1,      # 添加前的泵1体积  
            total_volume=current_total_vol,  # 添加前的总体积
            action=action,
            volume_added=signed_vol,
            mix_time=mix_time,
            next_ph=new_ph,              # 添加后的pH
            next_pump0_ml=self.pump0_volume,  # 添加后的泵0体积
            next_pump1_ml=self.pump1_volume,  # 添加后的泵1体积
            next_total_volume=self.current_volume  # 添加后的总体积
        )
        
        return new_ph, direction

    def _select_next_volume(self, volumes, mu, std, current_ph):
        """标准的期望改进(EI)采集函数 - 与论文描述一致"""
        
        # 计算预测误差和当前最佳误差
        predicted_errors = np.abs(mu - self.target_ph)
        current_best_error = min([abs(ph - self.target_ph) for ph in self.ph_values])
        
        # EI核心计算
        imp = current_best_error - (predicted_errors - 0.01)  # 改进量
        Z = imp / std  # 标准化
        Z = np.where(std == 0, 0, Z)  # 处理零标准差情况
        
        # 标准EI公式
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std == 0.0] = 0.0
        
        # 物理方向约束
        is_above_target = current_ph > self.target_ph
        if is_above_target:
            direction_valid = volumes <= self.current_volume  # 只能加酸
        else:
            direction_valid = volumes >= self.current_volume  # 只能加碱
        
        # 步长约束
        step_valid = (np.abs(volumes - self.current_volume) >= self.min_step) & \
                    (np.abs(volumes - self.current_volume) <= self.max_step)
        
        valid_mask = direction_valid & step_valid
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            # 如果没有有效点，返回最小步长
            step = self.min_step * (-1 if is_above_target else 1)
            return self.current_volume + step, "acid" if is_above_target else "base"
        
        # 选择EI值最大的点
        best_idx = valid_indices[np.argmax(ei[valid_indices])]
        best_direction = "acid" if is_above_target else "base"
        
        return volumes[best_idx], best_direction

    # ---------- 微调 ----------
    def _fine_tune_model(self):
        n = len(self.labeled_data)
        if n < 2:
            return
        ds = TitrationDataset(
            [{"comps": [], "global": np.array([0.1, 2.0, 298.15]),
              "v": v, "ph": ph} for v, ph in self.labeled_data],
            scaler=self.scaler)
        loader = DataLoader(ds, batch_size=min(32, n), shuffle=True, drop_last=False)
        optimizer = torch.optim.AdamW(self.model.fusion.parameters(),
                                      lr=self.lr_fine, weight_decay=1e-4)
        self.model.train()
        for _ in range(min(self.epochs_fine, max(5, n))):
            for batch in loader:
                x_sp = torch.zeros(batch["x_species"].shape[0], 3, 4).to(self.device)
                x_gl = batch["x_global"].to(self.device)
                v_t = batch["v"].to(self.device)
                ph_t = batch["ph"].to(self.device)
                ph_p = self.model(x_sp, x_gl, v_t)
                loss = torch.nn.functional.mse_loss(ph_p, ph_t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        torch.save({
            "encoder": self.model.encoder.state_dict(),
            "vol_enc": self.model.vol_enc.state_dict(),
            "fusion": self.model.fusion.state_dict(),
            "scaler": self.scaler
        }, "ckpt/finetuned_latest.pt")

    # ---------- 绘图 ----------
    def show_plot(self):
        if not self.labeled_data:
            return
        v_meas, ph_meas = zip(*self.labeled_data)
        v_grid = np.arange(0, max(v_meas) + 5, 0.1)
        if len(self.labeled_data) >= 2:
            X = np.array([[v] for v in v_meas])
            y = np.array(ph_meas)
            self.gp.fit(X, y)
            mu = self.gp.predict(v_grid.reshape(-1, 1))
        else:
            mu = np.full_like(v_grid, ph_meas[0])
        plt.figure(figsize=(8, 4))
        plt.plot(v_grid, mu, label='GP 预测曲线')
        plt.scatter(v_meas, ph_meas, color='red', label='实测点')
        plt.axhline(self.target_ph, color='green', linestyle='--', label='目标 pH')
        plt.xlabel("累计体积 (mL)")
        plt.ylabel("pH")
        plt.title("主动学习滴定曲线")
        plt.legend()
        plt.savefig("active_titration.png", dpi=300)
        plt.close()

    # ---------- 主入口 ----------
    def run_learning(self):
        try:
            # 初始测量
            initial_ph, _ = self._measure_ph()
            self.labeled_data.append((0.0, initial_ph))
            
            hit_cnt = 0
            iteration_count = 0  # 添加迭代计数器

            for step in range(self.budget_real - 1):  # 减1确保不会超过预算
                iteration_count += 1  # 每次迭代增加计数
                print(f"🔁 第 {iteration_count} 次迭代开始")
                
                self._fine_tune_model()
                
                # 生成候选体积点
                v_grid = np.arange(max(0, self.current_volume - 10.0),
                                self.current_volume + 20.0, self.min_step)
                
                mu, std = self._predict_curve(v_grid)
                current_ph = self.labeled_data[-1][1]
                
                next_v, next_direction = self._select_next_volume(v_grid, mu, std, current_ph)
                
                # 计算需要添加的体积
                vol_add = abs(next_v - self.current_volume)
                
                # 执行滴定并获取新pH值
                new_ph, actual_direction = self._measure_ph(vol_add, direction=next_direction)
                
                new_volume = self.current_volume
                self.labeled_data.append((new_volume, new_ph))
                
                # 打印当前迭代结果
                print(f"第 {iteration_count} 次迭代结果: 体积={new_volume:.2f}ml, pH={new_ph:.2f}, 目标pH={self.target_ph:.2f}")
                
                # 命中检测
                if abs(new_ph - self.target_ph) < 0.1:
                    print(f"第 {iteration_count} 次迭代命中目标区间 ")
                    break
            
            # 在最终日志中添加迭代次数信息
            self._log_final_state()
            print(f"总共进行了 {iteration_count} 次迭代")
            self.show_plot()
            return self.labeled_data, self.current_volume, iteration_count  # 返回迭代次数

        except Exception as e:
            if not self.simulate:
                self.arduino.stop_all_pumps()
            print(f"❌ 主动学习在第 {iteration_count} 次迭代中断:", e)
            raise
    def _log_final_state(self):
        """记录最终状态"""
        if not self.labeled_data:
            return
        
        # 读取最终稳定pH值
        final_ph = self._read_stable_ph()
        
        # 记录最终状态
        self.logger.log_step(
            cur_ph=final_ph,                   # 最终pH
            target_ph=self.target_ph,
            pump0_ml=self.pump0_volume,        # 最终泵0体积
            pump1_ml=self.pump1_volume,        # 最终泵1体积  
            total_volume=self.current_volume,  # 最终总体积
            action="final",
            volume_added=0.0,
            mix_time=0.0,
            next_ph=final_ph,                  # 最终pH（与cur_ph相同）
            next_pump0_ml=self.pump0_volume,   # 最终泵0体积
            next_pump1_ml=self.pump1_volume,   # 最终泵1体积
            next_total_volume=self.current_volume  # 最终总体积
        )
    def _predict_curve(self, volumes):
        """返回 (mu, std)"""
        n = len(self.labeled_data)

        # 1) 0-2 条：纯预训练 NN
        if n <= 2:
            self._load_pretrained_weights()
            mu = self._predict_with_nn(volumes, fine_tuned=False)
            return mu, np.zeros_like(mu)   # 补一个 0-std

        # 2) 3 条：NN + GP 残差
        if n == 3:
            mu_nn = self._predict_with_nn(volumes, fine_tuned=False)
            # 训练 GP 学习残差
            X = np.array([[v] for v, ph in self.labeled_data])
            res = np.array([ph - self._predict_with_nn([v], fine_tuned=False)[0]
                for v, ph in self.labeled_data])
            self.gp.fit(X, res)
            mu_res, std = self.gp.predict(volumes.reshape(-1, 1), return_std=True)
            return mu_nn + mu_res, std

        # 3) ≥4 条：微调 NN + GP 残差
        else:
            mu_nn = self._predict_with_nn(volumes, fine_tuned=True)
            X = np.array([[v] for v, ph in self.labeled_data])
            res = np.array([ph - self._predict_with_nn([v], fine_tuned=False)[0]
                for v, ph in self.labeled_data])
            self.gp.fit(X, res)
            mu_res, std = self.gp.predict(volumes.reshape(-1, 1), return_std=True)
            return mu_nn + mu_res, std

    def _predict_with_nn(self, volumes, fine_tuned=True):
        self.model.eval()
        preds = []
        with torch.no_grad():
            for v in volumes:               # v 是标量
                x_sp = torch.zeros(1, 3, 4).to(self.device)
                x_gl = torch.tensor([[0.1, 2.0, 298.15]]).to(self.device)
                v_t = torch.tensor([[float(v)]]).to(self.device)  # ✅ 只有两层[]
                pred = self.model(x_sp, x_gl, v_t).cpu().item()
                preds.append(pred)
        return np.array(preds)

# ---------------------- 一键运行 ----------------------
if __name__ == "__main__":
    # 1. 模拟模式
    learner = ActiveTitrationLearner(
        arduino_controller=None,
        target_ph=7.0,
        simulate=True,
        csv_path=r"E:\mypython\SDL\all.csv"
    )
    curve, final_vol = learner.run_learning()
    print("最终体积:", final_vol, "mL")
    print("完整曲线已保存为 active_titration.png")