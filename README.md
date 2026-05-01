# Hemodynamic Blood Flow Simulator (1-D)

A high-performance Python simulation of pulsatile blood flow through a narrowed (stenosed) artery. This project models hemodynamic properties including velocity profiles, pressure gradients, and wall shear stress in stenosed arteries.

## 🚀 Quick Start

### 1. Clone the repository & enter the directory
```bash
git clone https://github.com/nikoswahlberg/hemodynamic-blood-flow-simulator.git
cd hemodynamic-blood-flow-simulator
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the simulation
```bash
python main.py
```

## ⚙️ Configuration

You can easily modify clinical scenarios by tweaking the constants at the top of `main.py`:

- **STEN_SEVERITY**: Change the narrowing percentage (e.g., 0.62 for 62% stenosis)
- **HEART_RATE**: Adjust the BPM (e.g., 120 for exercise)
- **MU**: Adjust blood viscosity to simulate conditions like anemia

## 📊 Features

- Real-time visualization of blood flow dynamics
- Configurable stenosis severity and vessel properties
- Pulsatile flow modeling with variable heart rates
- Analysis of pressure drops and wall shear stress

## 🛠 Tech Stack

- **Python 3.10+**
- **NumPy**: Numerical arrays and physics calculations
- **Matplotlib**: Real-time rendering and animation

## 📚 Documentation

For more detailed information about the physics and methodology, refer to the project documentation.

## 👤 Author

Created by **Nikos Wahlberg** — MSc Advanced Energy Solutions, Aalto University

## 📝 License

[Add appropriate license - MIT, Apache, etc.]
