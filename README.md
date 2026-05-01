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
