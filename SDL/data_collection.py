# titration_collector.py
import csv
import time
import argparse
from datetime import datetime
from to_arduino import ArduinoController   # 复用你的 Arduino 控制类

class TitrationCollector:
    def __init__(self,
                 port: str = 'COM3',
                 step_ml: float = 10,
                 step_wait: float = 15.0,
                 max_total_ml: float = 20.0,
                 use_pump0: bool = False):
        """
        :param port: 串口号
        :param step_ml: 每步滴定体积 (mL)
        :param step_wait: 每步等待时间 (s)，让 pH 稳定
        :param max_total_ml: 最大累计滴定体积 (mL)
        :param use_pump0: True=右泵(Pump0)加碱，False=左泵(Pump1)加酸
        """
        self.port = port
        self.step_ml = step_ml
        self.step_wait = step_wait
        self.max_total_ml = max_total_ml
        self.pump_id = 0 if use_pump0 else 1
        self.pump_name = "Pump0" if use_pump0 else "Pump1"

        # 初始化 Arduino
        self.arduino = ArduinoController(port)

        # CSV 文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_name = f"titration_{timestamp}.csv"
        with open(self.csv_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Cumulative_V_mL", "pH", "Timestamp"])

    def run(self):
        cumulative_ml = 0.0
        print("开始滴定...")
        print(f"数据将写入 {self.csv_name}")
        self.arduino.FAN_ON()
        self.arduino.HEAT_OFF()
        i = 0

        while cumulative_ml <= self.max_total_ml:
            # 读取当前 pH
            ph = self.arduino.read_ph()
            if i == 0:
                pH = ph
                i = 1
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if ph is None:
                ph = "NA"
            print(f"{ts}  V={cumulative_ml:.2f} mL  pH={ph}")

            # 记录
            if pH> 7:
                with open(self.csv_name, 'a', newline='') as f:
                    csv.writer(f).writerow([f"{-cumulative_ml:.2f}", ph, ts])
            else:
                with open(self.csv_name, 'a', newline='') as f:
                    csv.writer(f).writerow([f"{cumulative_ml:.2f}", ph, ts])
            # 滴定结束判断
            if cumulative_ml >= self.max_total_ml:
                print("已到达最大滴定体积，结束。")
                break

            # 发送滴定命令
            if self.pump_id == 0:
                self.arduino.pump0_flow(self.step_ml)
            else:
                self.arduino.pump1_flow(self.step_ml)

            cumulative_ml += self.step_ml
            time.sleep(10)

        # 清理
        self.arduino.close()
        print("滴定完成，串口已关闭。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动滴定数据收集脚本")
    parser.add_argument('--port', default='COM3', help='Arduino 串口号 (默认 COM3)')
    parser.add_argument('--step', type=float, default=0.2, help='每步滴定体积 (mL, 默认 0.1)')
    parser.add_argument('--wait', type=float, default=5.0, help='每步等待时间 (s, 默认 5)')
    parser.add_argument('--max', type=float, default=40.0, help='最大累计滴定体积 (mL, 默认 20)')
    parser.add_argument('--acid', action='store_true', help='使用左泵(Pump1)滴加酸液，默认右泵(Pump0)加碱')
    args = parser.parse_args()

    collector = TitrationCollector(
        port=args.port,
        step_ml=args.step,
        step_wait=args.wait,
        max_total_ml=args.max,
        use_pump0=True
    )
    collector.run()