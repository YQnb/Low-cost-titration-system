# Low-Cost Automated Titration Platform

A "simulation-to-reality" framework for intelligent pH control in complex buffered systems using low-cost hardware, physics-informed neural networks, and Bayesian active learning.

This project presents an intelligent, low-cost automated titration platform designed for rapid and precise pH regulation in complex buffered systems. Unlike traditional PID or black-box ML approaches, our system embeds physicochemical priors into a transferable neural surrogate model, and uses Gaussian-process-based active learning to close the loop between simulation and experiment.

The platform autonomously determines optimal titrant additions, corrects for model-reality mismatches, and converges to target pH values in as few as 3–4 iterations, without requiring historical experimental data.

## ✨ Features

- **Physics-Informed Surrogate Model**: A DeepSet-based neural network trained on theoretical titration curves generated from charge balance equations, capturing multi-buffer equilibria and ionic strength effects.
- **Hybrid Modeling**: Combines pretrained neural network with Gaussian Process residual learning to adapt to real-system deviations.
- **Target-Directed Expected Improvement (EI)**: A modified acquisition function that incorporates directional chemical constraints (e.g., acid/base only when needed) and step-size bounds.
- **Low-Cost & Modular**: Entire hardware setup costs ≈ $100, using Arduino, peristaltic pumps, and open-source components.
- **Generalizable**: Validated across phosphate, citrate, acetate, and ammonium buffer systems, including binary and multi-component mixtures.

## 🔬 Scientific Workflow

1. **Theoretical Prior Generation**
   Solve charge balance equations numerically to generate titration curves for diverse buffer systems → train physics-informed surrogate model.

2. **Transfer Learning to Real System**
   Load pretrained model → fine-tune with real-time experimental data via Gaussian Process residual modeling.

3. **Active Learning Loop**
   Use target-directed EI to select next titrant volume → execute via Arduino → update model → iterate until |pH − target| < 0.1.

## 📁 Project Layout
SDL-python/
├─ main.py # Tkinter-based pH control UI
├─ data_collection.py # CLI data recorder for titration curves
├─ to_arduino.py # Serial protocol helper for pumps/sensors
├─ activate_learn/ # Active learning agent + logging utilities
├─ model_pre_train/ # Pretrained titration model weights/code
├─ tiration_curves_predict/ # Notebooks/scripts for theoretical curve prediction
├─ train_csv/, logs/, ckpt/ # Data, run logs, pretrained checkpoints
└─ active_logs/, active_learn # Experiment metadata

text

## 🛠️ Arduino Firmware & Libraries

The Arduino firmware is located in the `SDL-arduino/` directory. This firmware handles all hardware communication and control.

### Required Arduino Libraries

- **Adafruit_TCS34725** - For RGB color sensor
- **DallasTemperature** - For temperature sensors (if used)
- **OneWire** - For 1-wire communication
- **AccelStepper** - For precise pump control (if using stepper motors)

### Arduino Code Structure
SDL-arduino/
├── titration_controller/ # Main Arduino firmware
│ ├── titration_controller.ino # Main sketch
│ ├── PumpController.h # Pump control logic
│ ├── SensorManager.h # pH and color sensor management
│ └── CommandParser.h # Serial command parsing
├── libraries/ # Custom libraries (if any)
└── calibration/ # Calibration utilities

text

### Key Arduino Commands

The Arduino responds to the following serial commands (defined in `to_arduino.py`):

- `READ_PH` - Read current pH value
- `READ_COLOR` - Read RGB color sensor values
- `PUMP0:<amount>` - Control pump 0 (base)
- `PUMP1:<amount>` - Control pump 1 (acid)
- `FAN_ON` / `FAN_OFF` - Stirring control
- `HEAT_ON` / `HEAT_OFF` - Temperature control
- `LED` - Toggle LED illumination
- `PUMPx:STARTCAL` - Start pump calibration
- `PUMPx:SETCAL:<value>` - Set pump calibration factor
- `STOP_ALL` - Emergency stop all pumps

### Flashing the Arduino

1. Open `SDL-arduino/titration_controller/titration_controller.ino` in Arduino IDE
2. Install required libraries via Library Manager
3. Select correct board and port
4. Upload the sketch

## 📋 Requirements

### Software Requirements
- Python 3.9+ (tested on Windows 10)
- Arduino IDE 2.x for firmware updates

### Hardware Requirements
- Arduino Uno (or compatible)
- Gravity Peristaltic Pumps (2×, acid/base, 0.1 mL accuracy)
- Analog pH Sensor (3-point calibrated)
- Magnetic Stirrer (custom-built or commercial)
- RGB Color Sensor (TCS34725)
- LED Module (for illumination)
- 3D-Printed Mounting Plate (modular design)

### Python Dependencies

```bash
pip install torch torchvision torchaudio
pip install numpy scipy pandas scikit-learn matplotlib
pip install pyserial pillow
(Adjust CUDA-enabled wheels as needed; see model_pre_train for additional dependencies.)

🚀 Getting Started
1. Hardware Setup
Wire components according to the pin definitions in SDL-arduino/titration_controller.ino

Flash Arduino with the provided firmware

Calibrate sensors - pH sensor requires 3-point calibration

2. Software Configuration
Configure serial port: edit the port argument in main.py, data_collection.py, or pass --port COMx when running scripts.

Prepare checkpoints: place pretrain_best.pt (and optional fine-tuned weights) into ckpt/.

💻 Usage
1. Run the Control GUI
powershell
cd SDL-python
python main.py --port COM3
Monitor live pH and color, issue manual pump commands, or enable "自动调节" to let ActiveTitrationLearner drive toward the target pH.

Pump calibration buttons trigger PUMPx:STARTCAL and accept measured mL/step so future dosing is accurate.

2. Collect Raw Titration Curves
powershell
python data_collection.py --port COM3 --step 0.2 --wait 5 --max 40
Generates titration_YYYYmmdd_HHMMSS.csv with signed cumulative volume (acid additions negative), pH, and timestamps.

Ideal for building datasets under train_csv/ or validating the active learner.

3. Active Learning-only Runs
powershell
python -m activate_learn.activate_learn_little \
  --target_ph 7.0 \
  --simulate  # optional, uses recorded CSV instead of hardware
In hardware mode, the learner alternates between pump0_flow/pump1_flow, waits for stability, retrains/fine-tunes TitrationModel, and saves plots + logs (active_titration.png, active_logs/*.csv).

In simulation mode, it interpolates from a reference CSV to test new strategies safely.

🔄 Recommended Workflow
Calibrate pumps via the GUI so ml inputs reflect physical volume.

Collect baseline curves using data_collection.py to validate sensors.

Pretrain / fine-tune the neural model with your CSVs (model_pre_train/ scripts).

Enable active learning to autonomously hit target pH values with fewer manual interventions.

Analyze logs (active_logs/) and generated plots to compare experiments or retrain models.

⚠️ Safety Notes
Always keep an emergency stop (physical switch) for pumps; the software issues STOP_ALL, but hardware should fail-safe.

Secure cables and avoid spills on the Arduino or PC.

Validate each firmware command manually before running unattended loops.

Use appropriate chemical containment for acid/base solutions.

🔧 Troubleshooting
Common Issues
Serial timeout / None pH: ensure the Arduino replies to READ_PH; check baud rate (115200) and cable connection.

GUI freeze: long-running serial calls should stay in threads; ensure color_interval stays ≥100 ms and sensors respond quickly.

Model errors (ckpt/pretrain_best.pt missing): run the pretraining pipeline or copy the provided checkpoint into ckpt/.

Active learning plateau: inspect active_logs/*.csv to confirm volumes/pH are recorded; recalibrate pumps if measured dosing drifts.

Pump calibration issues: verify pump wiring and check for mechanical obstructions.

Arduino-specific Issues
Pump not responding: check motor driver connections and power supply

Sensor readings erratic: verify wiring and ground connections

Serial communication drops: check baud rate consistency and cable quality

📄 License
Provide your preferred license terms here. If undecided, consider adding an MIT License file so collaborators understand permitted use.

🤝 Contributing
We welcome contributions! Please feel free to submit issues, feature requests, or pull requests to improve the platform.

For detailed technical specifications and circuit diagrams, refer to the SDL-arduino/ documentation.