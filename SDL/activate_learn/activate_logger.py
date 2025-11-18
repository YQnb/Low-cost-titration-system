import csv, os, json, time
from datetime import datetime
from pathlib import Path

class ActiveLearningLogger:
    """
    负责记录主动学习过程中的
    (t, pH, target_pH, pump0_ml, pump1_ml, action, reward, next_pH)
    每轮写入一行，文件按日期分片。
    """

    def __init__(self, log_dir: str = "active_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 当天文件
        today = datetime.now().strftime("%Y%m%d")
        self.csv_path = self.log_dir / f"active_learning_{today}.csv"

        # 若文件不存在则写表头
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "current_ph",
                    "target_ph",
                    "pump0_ml",
                    "pump1_ml",
                    "action",          # 0/1/2 或 'add_base'/'add_acid'/'hold'
                    "reward",          # 可选
                    "next_ph",         # 可选
                    "extra"            # JSON 字符串，可扩展
                ])

    def log_step(self, cur_ph, target_ph, pump0_ml, pump1_ml,
                 action, **kwargs):
        row = [
            datetime.now().isoformat(timespec="seconds"),
            cur_ph,
            target_ph,
            pump0_ml,
            pump1_ml,
            action,  #-1:加酸  1:加碱
            json.dumps(kwargs)
        ]
        # 线程安全：简单追加即可
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)