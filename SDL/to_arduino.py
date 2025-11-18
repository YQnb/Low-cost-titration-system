import serial
import time
from threading import Thread


class ArduinoController:
    def __init__(self, port, baudrate=115200):
        self.ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)  # 等待连接建立
        self.ph_value = None
        self.running = True
        self.read_thread = Thread(target=self._read_serial)
        self.read_thread.start()
        self.calibration_status = {"pump0": False, "pump1": False}
        self.color_value = None
    
    def _read_serial(self):
        while self.running:
            time.sleep(1)
            if self.ser.in_waiting > 0:
                self.ph_value = self.read_ph()
    
    def LED_contral(self):
        command = f"LED\n"
        self.ser.write(command.encode())

    def start_pump_calibration(self, pump_id):
        """开始泵的校准过程"""
        command = f"PUMP{pump_id}:STARTCAL\n"
        self.ser.write(command.encode())
    
    def set_pump_calibration(self, pump_id, quantification):
        """设置泵的校准值"""
        command = f"PUMP{pump_id}:SETCAL:{quantification}\n"
        self.ser.write(command.encode())
        self.calibration_status[f"pump{pump_id}"] = True
    
    def get_calibration_status(self, pump_id):
        """获取泵的校准状态"""
        return self.calibration_status[f"pump{pump_id}"]
    
    def read_ph(self):
        """读取pH值"""
        self.ser.write(b"READ_PH\n")
        time.sleep(0.1) 
        line = self.ser.readline().decode('utf-8').strip()
        if line.startswith("PH_VALUE:"):
            try:
                self.ph_value = float(line.split(":")[1])
            except (IndexError, ValueError):
                pass
        return self.ph_value   # 直接返回最新的ph_value
    
    def pump0_flow(self, amount):     #右侧泵  碱
        self.ser.write(f"PUMP0:{amount}\n".encode())
    
    def pump1_flow(self, amount):      #左侧泵 酸
        self.ser.write(f"PUMP1:{amount}\n".encode())

    def HEAT_ON(self):      
        self.ser.write(b"Heat_ON\n")

    def HEAT_OFF(self):      
        self.ser.write(b"Heat_OFF\n")
    
    def FAN_ON(self):      
        self.ser.write(b"FAN_ON\n")

    def FAN_OFF(self):      
        self.ser.write(b"FAN_OFF\n")

    def stop_all_pumps(self):
        self.ser.write(b"STOP_ALL\n")
    
    def close(self):
        self.running = False
        # self.read_thread.join()
        self.ser.close()

    def read_color(self):
        """读取颜色传感器值，返回RGB值"""
        self.ser.write(b"READ_COLOR\n")
        time.sleep(0.1)  # 等待Arduino响应
        
        # 初始化返回值
        r, g, b = 0, 0, 0
        
        # 读取串口缓冲区中的所有数据
        while self.ser.in_waiting > 0:
            line = self.ser.readline().decode('utf-8').strip()
            
            if line.startswith("COLOR_R:"):
                try:
                    r = int(line.split(":")[1])
                except (IndexError, ValueError):
                    pass
            elif line.startswith("COLOR_G:"):
                try:
                    g = int(line.split(":")[1])
                except (IndexError, ValueError):
                    pass
            elif line.startswith("COLOR_B:"):
                try:
                    b = int(line.split(":")[1])
                except (IndexError, ValueError):
                    pass
        self.color_value = [r, g, b]
        return self.color_value