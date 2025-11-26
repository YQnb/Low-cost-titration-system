#include "GravityPump.h"
#include <EEPROM.h>

#define CALIBRATIONTIME 15      // when Calibration pump running time, unit second
#define ReceivedBufferLength 20

// EEPROM address calculation macros
#define FLOWRATE_BASE_ADDRESS 0x00    // Base address for flow rates
#define PUMPSPEED_BASE_ADDRESS 0x40   // Base address for pump speeds
#define PUMP_CONFIG_SIZE 8            // Size of each pump's config in bytes

#define EEPROM_write(address, p) {int i = 0; byte *pp = (byte*)&(p);for(; i < sizeof(p); i++) EEPROM.write(address+i, pp[i]);}
#define EEPROM_read(address, p)  {int i = 0; byte *pp = (byte*)&(p);for(; i < sizeof(p); i++) pp[i]=EEPROM.read(address+i);}

GravityPump::GravityPump() : 
    _pin(0),
    _pumpId(0),
    _runFlag(false),
    _stopFlag(false),
    _pumpSpeed(180),
    _flowRate(0.75),
    _startTime(0),
    _intervalTime(0)
{
}

GravityPump::~GravityPump()
{
}

void GravityPump::setPin(int pin)   // pump pin setting
{
    this->_pin = pin;
    this->_pumpServo.attach(this->_pin);
}

void GravityPump::setPumpId(int id) // set unique ID for each pump
{
    this->_pumpId = id;
}

void GravityPump::getFlowRateAndSpeed()      // flowrate and speed reading from EEPROM
{
    // Calculate address based on pump ID
    int flowRateAddress = FLOWRATE_BASE_ADDRESS + (this->_pumpId * PUMP_CONFIG_SIZE);
    int pumpSpeedAddress = PUMPSPEED_BASE_ADDRESS + (this->_pumpId * PUMP_CONFIG_SIZE);
    
    EEPROM_read(flowRateAddress, this->_flowRate);
    delay(5);
    EEPROM_read(pumpSpeedAddress, this->_pumpSpeed);
}

void GravityPump::update()      // get the state from system, need to be put in the loop.
{
    pumpDriver(this->_pumpSpeed, this->_intervalTime);
}

void GravityPump::pumpDriver(int speed, unsigned long runTime)      // the basic pump function
{
    if(this->_stopFlag || millis() - this->_startTime >= runTime)
    {
        this->_runFlag = false;
        this->_stopFlag = false;
        this->_pumpServo.write(this->_servoStop);
        this->_startTime = millis() + runTime;
    }
    else
    {
        this->_pumpServo.write(speed);
    }    
}

float GravityPump::flowPump(float quantitation,int speed)     // quantification setting pump function
{
    unsigned long runTime = 0;
    if(!this->_runFlag)
    {
        this->_pumpSpeed = speed;
        this->_runFlag = true;
        this->_intervalTime = 1000*(quantitation / this->_flowRate);
        this->_startTime = millis();
        return this->_intervalTime; 
    }
    return 0;
}

float GravityPump::timerPump(unsigned long runTime) // timer pump function
{
    if(!this->_runFlag)
    {
        this->_runFlag = true;
        this->_intervalTime = runTime*1000;
        this->_startTime = millis();
        return (this->_flowRate*runTime);
    }
    return 0;
}

void GravityPump::stop()    // stop function
{
    this->_stopFlag = true;
    this->_runFlag = false;
}

void GravityPump::calFlowRate(int speed) {
    this->_pumpSpeed = speed;
    this->_stopFlag = false;
    this->_startTime = millis();
    this->_intervalTime = (CALIBRATIONTIME*1000);
    Serial.println(F("Calibration starting..."));
}

void GravityPump::setCalibrationValue(float quantification)
{
    Serial.print(F("Quantification:"));
    Serial.println(quantification);
    
    // 修正流量率计算公式
    this->_flowRate = quantification/float(CALIBRATIONTIME);  // 直接使用输入值作为流量率(ml/s)
    
    // Calculate address based on pump ID
    int flowRateAddress = FLOWRATE_BASE_ADDRESS + (this->_pumpId * PUMP_CONFIG_SIZE);
    int pumpSpeedAddress = PUMPSPEED_BASE_ADDRESS + (this->_pumpId * PUMP_CONFIG_SIZE);
    
    EEPROM_write(flowRateAddress, this->_flowRate);
    delay(10);
    EEPROM_write(pumpSpeedAddress, this->_pumpSpeed);
    
    Serial.print(F("Pump ID: "));
    Serial.println(this->_pumpId);
    Serial.print(F("PumpSpeed: "));
    Serial.println(this->_pumpSpeed);
    Serial.print(F("FlowRate: "));
    Serial.print(this->_flowRate, 4);  // 显示4位小数
    Serial.println(F("ml/s,\r\nCalibration Finish!"));
}
void GravityPump::initServo() {
    if (_pin != 0) {
        _pumpServo.attach(_pin);
        _pumpServo.write(_servoStop);  // 设置为停止状态
    }
}