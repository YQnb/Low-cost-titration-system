
from threading import Thread
import sys
import tkinter as tk
import csv
from datetime import datetime
from to_arduino import ArduinoController
import time
import threading
from queue import Queue
from activate_learn.activate_learn_little import ActiveTitrationLearner

class PumpControlApp:
    def __init__(self, arduino, activate):
        self.arduino = arduino
        self.logger = DataLogger(arduino)  # 创建数据记录器
        self.root = tk.Tk()
        self.root.title("pH 控制系统")
        self.data_queue = Queue(maxsize=1)  # 只保留最新数据
        self.running = True
        self.data_thread = threading.Thread(target=self._data_reader, daemon=True)
        self.data_thread.start()
        self.activate = activate
        
        
        # 创建主框架
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(expand=True, fill='both')
        
        # pH 显示区域
        ph_frame = tk.LabelFrame(main_frame, text="pH 监控", padx=10, pady=5)
        ph_frame.pack(fill='x', pady=5)
        
        # 当前pH值显示
        self.ph_label = tk.Label(ph_frame, text=f"当前 pH: {self.arduino.ph_value}", font=('Arial', 24))
        self.ph_label.pack(side='left', padx=10)
        
        # pH目标值设置
        target_frame = tk.Frame(ph_frame)
        target_frame.pack(side='right', padx=10)
        
        tk.Label(target_frame, text="目标 pH:").pack(side='left')
        self.target_ph = tk.Entry(target_frame, width=6)
        self.target_ph.insert(0, "7.0")
        self.target_ph.pack(side='left', padx=5)

        # ===== 新增颜色传感器区域 =====
        color_frame = tk.LabelFrame(main_frame, text="颜色传感器控制", padx=10, pady=5)
        color_frame.pack(fill='x', pady=5)
        
        # 第一行：状态显示
        status_row = tk.Frame(color_frame)
        status_row.pack(fill='x', pady=2)
        
        self.color_status = tk.Label(
            status_row,
            text="状态: 未校准",
            fg="orange",
            font=('Arial', 10)
        )
        self.color_status.pack(side='left', padx=5)
        
        self.color_value_label = tk.Label(
            status_row,
            text="RGB: (---, ---, ---)",
            font=('Arial', 10)
        )
        self.color_value_label.pack(side='left', padx=10)
        
        # 颜色可视化显示
        self.color_display = tk.Label(
            status_row,
            text="    ",
            relief='sunken',
            width=8
        )
        self.color_display.pack(side='right', padx=5)
        
        # 第二行：校准控制
        calib_row = tk.Frame(color_frame)
        calib_row.pack(fill='x', pady=2)
        
        tk.Button(
            calib_row,
            text="白平衡校准",
            command=lambda: self.calibrate_color_sensor('white'),
            width=12
        ).pack(side='left', padx=5)
        
        tk.Button(
            calib_row,
            text="LED",
            command=self.arduino.LED_contral,
            width=8
        ).pack(side='right', padx=5)
        
        # 第三行：高级设置
        adv_row = tk.Frame(color_frame)
        adv_row.pack(fill='x', pady=2)
        
        tk.Label(adv_row, text="采样间隔(ms):").pack(side='left')
        self.color_interval = tk.Entry(adv_row, width=6)
        self.color_interval.insert(0, "1000")
        self.color_interval.pack(side='left', padx=5)
        
        tk.Checkbutton(
            adv_row,
            text="自动记录",
            variable=tk.BooleanVar(value=True)
        ).pack(side='right', padx=5)
        
        # 初始化颜色显示
        self.update_data()
        
        # 泵校准区域
        cal_frame = tk.LabelFrame(main_frame, text="泵校准", padx=10, pady=5)
        cal_frame.pack(fill='x', pady=5)
        
        # 泵0校准
        pump0_cal_frame = tk.Frame(cal_frame)
        pump0_cal_frame.pack(fill='x', pady=2)
        tk.Label(pump0_cal_frame, text="右泵校准:").pack(side='left')
        self.pump0_cal_btn = tk.Button(pump0_cal_frame, text="开始校准", 
                                     command=lambda: self.start_calibration(0))
        self.pump0_cal_btn.pack(side='left', padx=5)
        self.pump0_quant = tk.Entry(pump0_cal_frame, width=6)
        self.pump0_quant.pack(side='left', padx=5)
        tk.Button(pump0_cal_frame, text="设置校准值", 
                 command=lambda: self.set_calibration(0)).pack(side='left', padx=5)
        
        # 泵1校准
        pump1_cal_frame = tk.Frame(cal_frame)
        pump1_cal_frame.pack(fill='x', pady=2)
        tk.Label(pump1_cal_frame, text="左泵校准:").pack(side='left')
        self.pump1_cal_btn = tk.Button(pump1_cal_frame, text="开始校准", 
                                     command=lambda: self.start_calibration(1))
        self.pump1_cal_btn.pack(side='left', padx=5)
        self.pump1_quant = tk.Entry(pump1_cal_frame, width=6)
        self.pump1_quant.pack(side='left', padx=5)
        tk.Button(pump1_cal_frame, text="设置校准值", 
                 command=lambda: self.set_calibration(1)).pack(side='left', padx=5)
        
        # 自动调节控制
        control_frame = tk.LabelFrame(main_frame, text="自动调节控制", padx=10, pady=5)
        control_frame.pack(fill='x', pady=5)
        
        self.auto_regulate_var = tk.BooleanVar(value=False)
        self.auto_regulate_btn = tk.Checkbutton(
            control_frame,
            text="启用自动调节",
            variable=self.auto_regulate_var,
            command=self.toggle_auto_regulate
        )
        self.auto_regulate_btn.pack(side='left', padx=10)
        
        # 泵控制区域
        pump_frame = tk.LabelFrame(main_frame, text="手动泵控制", padx=10, pady=5)
        pump_frame.pack(fill='x', pady=5)
        
        # 泵0控制
        pump0_frame = tk.Frame(pump_frame)
        pump0_frame.pack(fill='x', pady=2)
        tk.Label(pump0_frame, text="泵右:").pack(side='left')
        self.pump0_amount = tk.Entry(pump0_frame, width=6)
        self.pump0_amount.pack(side='left', padx=5)
        tk.Button(pump0_frame, text="启动", command=self.start_pump0).pack(side='left', padx=5)
        
        # 泵1控制
        pump1_frame = tk.Frame(pump_frame)
        pump1_frame.pack(fill='x', pady=2)
        tk.Label(pump1_frame, text="泵左:").pack(side='left')
        self.pump1_amount = tk.Entry(pump1_frame, width=6)
        self.pump1_amount.pack(side='left', padx=5)
        tk.Button(pump1_frame, text="启动", command=self.start_pump1).pack(side='left', padx=5)
        
        # 停止按钮
        tk.Button(pump_frame, text="停止所有泵", command=self.stop_pumps).pack(pady=5)
        
        # 状态显示
        self.status_label = tk.Label(main_frame, text="系统就绪", fg="green")
        self.status_label.pack(pady=5)

    def calibrate_color_sensor(self, mode='white'):
        """执行颜色传感器校准"""
        try:
            self.status_label.config(text=f"正在进行{mode}平衡校准...", fg="blue")
            self.root.update()
            
            # 发送校准命令到Arduino
            command = f"CALIB_{mode.upper()}\n".encode()
            self.arduino.ser.write(command)
            
            # 设置10秒后更新状态的定时器
            self.root.after(10000, lambda: self._update_calibration_status(mode))
            
        except Exception as e:
            self.color_status.config(text="状态: 校准失败", fg="red")
            self.status_label.config(text=f"校准失败: {str(e)}", fg="red")

    def _update_calibration_status(self, mode):
        """校准完成后更新状态"""
        try:
            # 检查校准是否成功（这里可以添加实际检查逻辑）
            # 假设校准总是成功
            self.color_status.config(
                text=f"状态: 已校准({mode})", 
                fg="green"
            )
            self.status_label.config(text=f"{mode}平衡校准完成", fg="green")
            
            # 立即更新显示
            self.update_data()
            
        except Exception as e:
            self.color_status.config(text="状态: 校准失败", fg="red")
            self.status_label.config(text=f"校准后状态更新失败: {str(e)}", fg="red")

    def _data_reader(self):
        """后台线程持续读取数据"""
        while self.running:
            # 读取pH值
            ph = self.arduino.read_ph()
            # 读取颜色值
            color = self.arduino.read_color()
            
            # 放入队列(如果队列满则丢弃旧数据)
            if self.data_queue.full():
                self.data_queue.get_nowait()
            self.data_queue.put((ph, color))
            time.sleep(5)  # 控制读取频率

    def update_data(self):
        """非阻塞UI更新"""
        try:
            # 获取最新数据(非阻塞)
            
            if not self.data_queue.empty():
                ph, color = self.data_queue.get_nowait()
                
                # 更新pH显示
                if ph is not None:
                    self.ph_label.config(text=f"当前 pH: {ph:.2f}")

                # 更新颜色显示
                if color:
                    r, g, b = color[0], color[1], color[2]
                    self._update_color_display(r, g, b)
                
                # 记录数据
                # self.logger.log_data(ph, color)
        
        except Exception as e:
            print(f"UI更新错误: {e}")
        
        # 设置下一次更新
        interval = max(100, int(self.color_interval.get()))  # 最小100ms间隔
        self.root.after(interval, self.update_data)

    def _update_color_display(self, r, g, b):
        """高效更新颜色显示"""
        r = min(255, max(0, r))
        g = min(255, max(0, g))
        b = min(255, max(0, b))
        # 使用StringVar减少配置调用
        if not hasattr(self, 'color_text_var'):
            self.color_text_var = tk.StringVar()
            self.color_value_label.config(textvariable=self.color_text_var)
        
        self.color_text_var.set(f"RGB: ({r:3d}, {g:3d}, {b:3d})")
        
        # 直接修改颜色显示
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        self.color_display.config(
            bg=hex_color,
            fg="white" if (r+g+b) < 384 else "black"
        )

    def start_calibration(self, pump_id):
        """开始泵的校准过程"""
        try:
            self.arduino.start_pump_calibration(pump_id)
            self.status_label.config(text=f"泵{pump_id}校准开始...", fg="blue")
        except Exception as e:
            self.status_label.config(text=f"校准启动失败: {str(e)}", fg="red")
    
    def set_calibration(self, pump_id):
        """设置泵的校准值"""
        try:
            if pump_id == 0:
                quantification = float(self.pump0_quant.get())
            if pump_id == 1:
                quantification = float(self.pump1_quant.get())
            
            self.arduino.set_pump_calibration(pump_id, quantification)
            self.status_label.config(text=f"泵{pump_id}校准值已设置", fg="green")
        except ValueError:
            self.status_label.config(text="请输入有效的校准值", fg="red")
        except Exception as e:
            self.status_label.config(text=f"校准设置失败: {str(e)}", fg="red")
    
    def toggle_auto_regulate(self):
        """主动学习"""
        if self.auto_regulate_var.get():
            try:
                target = float(self.target_ph.get())
                self.status_label.config(text=f"主动学习启动 - 目标pH: {target}", fg="blue")
                threading.Thread(target=self._run_active_learning, daemon=True).start()
            except ValueError:
                self.auto_regulate_var.set(False)
                self.status_label.config(text="目标pH值无效", fg="red")
        else:
            self.status_label.config(text="自动调节已禁用", fg="green")

    def _run_active_learning(self):
        """执行主动学习过程"""
        self.activate.run_learning()
    
    def start_pump0(self):
        try:
            amount = float(self.pump0_amount.get())
            self.arduino.pump0_flow(amount)
            self.logger.log_pump_operation(0, amount)  # 记录泵0操作
            self.status_label.config(text=f"泵0运行中 - 流量: {amount}ml", fg="blue")
        except ValueError:
            self.status_label.config(text="请输入有效的流量值", fg="red")
    
    def start_pump1(self):
        try:
            amount = float(self.pump1_amount.get())
            self.arduino.pump1_flow(amount)
            self.logger.log_pump_operation(1, amount)  # 记录泵1操作
            self.status_label.config(text=f"泵1运行中 - 流量: {amount}ml", fg="blue")
        except ValueError:
            self.status_label.config(text="请输入有效的流量值", fg="red")
    
    def stop_pumps(self):
        self.arduino.stop_all_pumps()
        self.arduino.FAN_ON()
        self.status_label.config(text="所有泵已停止", fg="green")
    
    def run(self):
        self.root.mainloop()

class DataLogger:
    def __init__(self, arduino):
        self.arduino = arduino
        self.filename = f"ph_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.pump0_volume = 0.0  # 泵0累计体积
        self.pump1_volume = 0.0  # 泵1累计体积
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'pH Value', 'Pump0 Volume(ml)', 'Pump1 Volume(ml)', 'Red', 'Green', 'Blue'])
    
    def log_data(self, ph, color):
        """非阻塞数据记录"""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            r, g, b = color if color else (None, None, None)
            
            # 使用线程执行文件写入
            def write_to_file():
                with open(self.filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        timestamp, 
                        ph, 
                        self.pump0_volume, 
                        self.pump1_volume, 
                        r, g, b
                    ])
            
            threading.Thread(target=write_to_file, daemon=True).start()
            
        except Exception as e:
            print(f"数据记录错误: {e}")
    
    def log_pump_operation(self, pump_id, amount):
        """记录泵操作"""
        if pump_id == 0:
            self.pump0_volume += amount
        elif pump_id == 1:
            self.pump1_volume += amount

class PhControlSystem:
    def __init__(self, port='COM3'):
        # 初始化所有组件
        self.arduino = ArduinoController(port)
        self.activate_learn = ActiveTitrationLearner(self.arduino)
        self.app = PumpControlApp(self.arduino, self.activate_learn)
        
    def run(self):
        """运行整个系统"""
        self.app.run()
        
    def cleanup(self):
        """清理资源"""
        self.arduino.close()

if __name__ == "__main__":
    # 创建pH控制系统实例
    system = PhControlSystem(port='COM3')
    
    try:
        # 运行系统
        system.run()
    except KeyboardInterrupt:
        print("系统正在关闭...")
    finally:
        # 确保资源被正确清理
        system.cleanup()