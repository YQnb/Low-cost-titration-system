#include "GravityPump.h"
#include "Button.h"

//================= 宏定义 =================
#define SENSOR_PIN      A3      // pH 传感器模拟输入引脚
#define OFFSET          -0.2   // pH 值偏差补偿
#define Heat            13      // 加热模块
#define SAMPLING_INTERVAL 20    // pH 采样间隔(ms)
#define ARRAY_LENGTH      40    // pH 采样数组长度
#define CALIBRATION_TIME  30    // 校准时间(秒)
#define ERROR_THRESHOLD   0.5   // pH 误差阈值
#define FAN 8                   // 搅拌模块

// 颜色传感器引脚
#define S0  3
#define S1  4
#define S2  5
#define S3  6
#define OUT 2

//================= 对象与变量 =================
GravityPump pump0;   // ID=0, Pin=9
GravityPump pump1;   // ID=1, Pin=10

int   pHArray[ARRAY_LENGTH];
int   pHArrayIndex = 0;
float lastpHValue = 0.0;
bool  isCalibrating = false;
unsigned long calibrationStartTime = 0;

// 颜色传感器
byte  countR = 0, countG = 0, countB = 0;
float redCalibrationFactor   = 1.0;
float greenCalibrationFactor = 1.0;
float blueCalibrationFactor  = 1.0;

unsigned long lastColorRead = 0;
const  unsigned long colorInterval = 100;  // 100 ms 轮询一次

int ledPin = 12;

//================= 函数声明 =================
double averageArray(int* arr, int number);
void   handleSerialCommand();
void   updatepHValue();
void   calibrate(bool isWhiteBalance);
void   startMeasurement();
void   updateColorSensor();

//================= setup =================
void setup() {
  // 泵初始化
  pump0.setPumpId(0); pump0.setPin(9);  pump0.initServo();
  pump1.setPumpId(1); pump1.setPin(10); pump1.initServo();

  Serial.begin(115200);
  while (!Serial);

  // 颜色传感器引脚
  pinMode(S0, OUTPUT); pinMode(S1, OUTPUT);
  pinMode(S2, OUTPUT); pinMode(S3, OUTPUT);
  pinMode(OUT, INPUT);
  pinMode(Heat, OUTPUT); pinMode(FAN, OUTPUT);
  digitalWrite(S0, HIGH); digitalWrite(S1, HIGH);
  digitalWrite(ledPin, HIGH);

  startMeasurement();
}

//================= loop =================
void loop() {
  pump0.update();
  pump1.update();
  updatepHValue();
  updateColorSensor();   // 轮询颜色传感器
  handleSerialCommand();

  if (isCalibrating) {
    if (millis() - calibrationStartTime >= CALIBRATION_TIME * 1000) {
      isCalibrating = false;
      Serial.println("Calibration Finish!");
    }
  }
}

//================= 颜色传感器轮询 =================
static byte colorFlag = 0;
void updateColorSensor() {
  static byte pulseCounter = 0;
  static unsigned long lastEdgeTime = 0;

  // 在 OUT 引脚出现下降沿时计数
  bool outState = digitalRead(OUT);
  static bool lastOutState = outState;
  if (outState != lastOutState) {
    lastOutState = outState;
    if (outState == LOW) {              // 下降沿
      pulseCounter++;
    }
  }

  // 每 colorInterval 更新一次颜色通道
  static unsigned long lastSwitch = 0;
  if (millis() - lastSwitch >= colorInterval) {
    lastSwitch = millis();

    switch (colorFlag) {
      case 0:
        countR = min(255, (int)(pulseCounter * redCalibrationFactor));
        digitalWrite(S2, HIGH); digitalWrite(S3, HIGH);
        break;
      case 1:
        countG = min(255, (int)(pulseCounter * greenCalibrationFactor));
        digitalWrite(S2, LOW);  digitalWrite(S3, HIGH);
        break;
      case 2:
        countB = min(255, (int)(pulseCounter * blueCalibrationFactor));
        digitalWrite(S2, LOW);  digitalWrite(S3, LOW);
        break;
    }

    colorFlag = (colorFlag + 1) % 3;
    pulseCounter = 0;
  }
}

//================= 其余函数 =================
void startMeasurement() {
  // 只做引脚初始化，后续由轮询完成
}

void calibrate(bool isWhiteBalance) {
  Serial.println("白平衡校准：请放置纯白物体...");
  digitalWrite(ledPin, HIGH);
  delay(10000);

  float sumR = 0, sumG = 0, sumB = 0;
  const int samples = 20;
  for (int i = 0; i < samples; i++) {
    delay(300);
    sumR += countR;
    sumG += countG;
    sumB += countB;
  }

  float avgR = sumR / samples;
  float avgG = sumG / samples;
  float avgB = sumB / samples;

  Serial.print("原始平均值 R:"); Serial.print(avgR);
  Serial.print(" G:"); Serial.print(avgG);
  Serial.print(" B:"); Serial.println(avgB);

  float maxVal = 255;
  redCalibrationFactor   *= maxVal / avgR;
  greenCalibrationFactor *= maxVal / avgG;
  blueCalibrationFactor  *= maxVal / avgB;

  Serial.println("白平衡校准完成");
}

void updatepHValue() {
  static unsigned long lastSample = 0;
  if (millis() - lastSample >= SAMPLING_INTERVAL) {
    pHArray[pHArrayIndex] = analogRead(SENSOR_PIN);
    pHArrayIndex = (pHArrayIndex + 1) % ARRAY_LENGTH;

    float voltage = averageArray(pHArray, ARRAY_LENGTH) * 5.0 / 1024;
    lastpHValue = 3.5 * voltage + OFFSET;
    lastSample = millis();
  }
}

//================= 串口命令解析 =================
void handleSerialCommand() {
  if (!Serial.available()) return;

  String cmd = Serial.readStringUntil('\n');
  cmd.trim();

  if (cmd == "READ_PH") {
    Serial.print("PH_VALUE:"); Serial.println(lastpHValue, 2);
  }
  else if (cmd == "READ_COLOR") {
    Serial.print("COLOR_R:"); Serial.println(countR);
    Serial.print("COLOR_G:"); Serial.println(countG);
    Serial.print("COLOR_B:"); Serial.println(countB);
  }
  else if (cmd == "CALIB_WHITE") {
    calibrate(true);
  }
  else if (cmd == "CALIB_BLACK") {
    calibrate(false);
  }
  else if (cmd.startsWith("PUMP0:")) {
    if (cmd == "PUMP0:STARTCAL") {
      pump0.calFlowRate(180);
    } else if (cmd.startsWith("PUMP0:SETCAL:")) {
      float q = cmd.substring(13).toFloat();
      pump0.setCalibrationValue(q);
    } else {
      float amt = cmd.substring(6).toFloat();
      if (amt >= 0 && amt <= 100) pump0.flowPump(amt,180);
    }
  }
  else if (cmd.startsWith("PUMP1:")) {
    if (cmd == "PUMP1:STARTCAL") {
      pump1.calFlowRate(0);
    } else if (cmd.startsWith("PUMP1:SETCAL:")) {
      float q = cmd.substring(13).toFloat();
      pump1.setCalibrationValue(q);
    } else {
      float amt = cmd.substring(6).toFloat();
      if (amt >= 0 && amt <= 100) pump1.flowPump(amt,0);
    }
  }
  else if (cmd == "STOP_ALL") {
    pump0.stop(); pump1.stop(); Serial.println("ALL_PUMPS_STOPPED");
  }
  else if (cmd == "Heat_ON")  digitalWrite(Heat, HIGH);
  else if (cmd == "Heat_OFF") digitalWrite(Heat, LOW);
  else if (cmd == "FAN_ON")   digitalWrite(FAN, HIGH);
  else if (cmd == "FAN_OFF")  digitalWrite(FAN, LOW);
}

//================= 工具函数 =================
double averageArray(int* arr, int number) {
  if (number <= 0) return 0;

  long sum = 0;
  for (int i = 0; i < number; i++) sum += arr[i];
  return (double)sum / number;
}