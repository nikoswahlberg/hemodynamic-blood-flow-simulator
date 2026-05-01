# Hemodynamic Blood Flow Simulator (1-D)

A high-performance Python simulation of pulsatile blood flow through a narrowed (stenosed) artery. This project models cardiovascular physics in real-time, visualizing how clinical stenoses impact blood velocity and pressure gradients.

![Simulation Demo](demo.gif) *(Note: Replace with a GIF of your simulation)*

## ✨ Features
* **Real-Time Physics:** Solves 1-D continuity, Bernoulli, and Hagen-Poiseuille equations.
* **Pulsatile Dynamics:** Simulates a realistic cardiac cycle (72 BPM).
* **Particle Advection:** Animates 180+ red blood cell proxies accelerating through the narrowing (Venturi effect).
* **Medical UI:** Dark-themed dashboard tracking instantaneous flow rate, peak velocity, and total pressure drop (ΔP).

## 🚀 Quick Start

**1. Clone the repository & enter the directory**
```bash
git clone [https://github.com/yourusername/bloodflow_miniproject.git](https://github.com/yourusername/bloodflow_miniproject.git)
cd bloodflow_miniproject

**2. Install dependencies
pip install -r requirements.txt

**3. Run the simulation
python main.py


⚙️ Configuration
You can easily modify clinical scenarios by tweaking the constants at the top of main.py:

STEN_SEVERITY: Change the narrowing percentage (e.g., 0.62 for 62% stenosis).

HEART_RATE: Adjust the BPM (e.g., 120 for exercise).

MU: Adjust blood viscosity to simulate conditions like anemia.

🛠 Tech Stack
Python 3.10+

NumPy: Numerical arrays and physics calculations

Matplotlib: Real-time rendering and FuncAnimation
