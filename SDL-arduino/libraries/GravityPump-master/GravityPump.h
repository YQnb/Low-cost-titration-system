#ifndef _GRAVITYPUMP_H_
#define _GRAVITYPUMP_H_
#include <Servo.h>
#include <Arduino.h>

class GravityPump
{
public:
    GravityPump();
    ~GravityPump();

    void update();                          // get the state from system, need to be put in the loop.
    void setPin(int pin);  
    void setPumpId(int id);                 // set the pin for GravityPump.
    void calFlowRate(int speed = 180); // 直接开始校准
    void setCalibrationValue(float quantification); // 直接设置校准值
    
    void pumpDriver(int speed, unsigned long runTime); // the basic pump function
    float timerPump(unsigned long runTime);  // timer pump function
    float flowPump(float quantitation, int speed);      // quantification setting pump function
    void getFlowRateAndSpeed();              // flowrate and speed reading from EEPROM
    void stop();                             // stop function
    void initServo();  // 

private:
    Servo _pumpServo;
    int _pin;
    int _pumpId;
    bool _runFlag;
    bool _stopFlag;
    int _pumpSpeed;
    float _flowRate;
    unsigned long _startTime;
    unsigned long _intervalTime;
    const int _servoStop = 90;
};

#endif