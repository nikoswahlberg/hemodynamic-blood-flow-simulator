# 🫀 Hemodynamic Blood Flow Simulator (1D)

A physics-based simulation of **pulsatile blood flow through a stenosed artery**, implemented in Python using the scientific stack (`numpy`, `matplotlib`).

This project visualizes how **velocity and pressure evolve along an artery with a narrowing (stenosis)**, combining fundamental fluid dynamics equations with an animated particle system representing red blood cells.

---

## 📌 Features

- 1D hemodynamic model of arterial blood flow  
- Gaussian-shaped stenosis (plaque narrowing)  
- Pulsatile cardiac waveform  
- Real-time animation:
  - Artery geometry  
  - RBC particle motion  
  - Velocity profile  
  - Pressure profile  
- Dark-themed scientific visualization  
- Physically interpretable outputs (velocity spikes, pressure drops)

---

## ⚙️ Physics Model

The simulation combines three core principles:

### 1. Continuity Equation (Mass Conservation)

\[
A(x)\cdot v(x,t) = Q(t)
\]

### 2. Bernoulli Equation (Energy Conservation)

\[
P + \frac{1}{2}\rho v^2 = \text{constant}
\]

### 3. Hagen–Poiseuille Law (Viscous Losses)

\[
\frac{dP}{dx} = -\frac{8\mu Q}{\pi R^4}
\]

### 4. Pulsatile Flow Input

\[
Q(t) = \bar{Q} + A \sin(\omega t)
\]

> ⚠️ Note: GitHub renders LaTeX only in certain contexts. If equations don’t render, they will still remain readable.

---

## 🧠 What This Simulates

- Acceleration of blood flow inside a stenosis  
- Pressure drop across the narrowing  
- Partial downstream pressure recovery  
- Time-varying flow due to the cardiac cycle (systole/diastole)

---

## 📦 Requirements

Install dependencies:

```bash
pip install numpy matplotlib
