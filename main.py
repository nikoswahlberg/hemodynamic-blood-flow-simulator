"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          HEMODYNAMIC BLOOD FLOW SIMULATOR — Stenosed Artery (1-D)            ║
║                                                                              ║
║  Physics:                                                                    ║
║  • Continuity  →  A(x)·v(x,t) = Q(t)           (mass conservation)           ║
║  • Bernoulli   →  P + ½ρv² = const              (inviscid energy balance)    ║
║  • Hagen-Poiseuille  →  dP/dx = -8μQ / (πR⁴)   (viscous pressure gradient)   ║
║  • Pulsatile waveform: Q(t) = Q̄ + Â·sin(ωt)                                  ║
║                                                                              ║
║  Dependencies: numpy, matplotlib  (standard scientific Python stack)         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ── Standard library ──────────────────────────────────────────────────────────
import sys

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 1 ── GLOBAL PHYSICAL & SIMULATION CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

# ── Blood fluid properties ────────────────────────────────────────────────────
RHO = 1060.0   # Blood density [kg/m³]
               # Slightly above water (1000) due to erythrocytes, plasma proteins.
               # Typical range: 1045–1065 kg/m³.

MU  = 0.0035   # Apparent (effective) dynamic viscosity [Pa·s]
               # Blood is non-Newtonian (shear-thinning, Carreau–Yasuda model),
               # but at physiological shear rates (~100 s⁻¹) a constant "apparent"
               # value of 0.003–0.004 Pa·s is a valid engineering approximation.

# ── Vessel geometry ───────────────────────────────────────────────────────────
L              = 0.10   # Arterial segment length [m]  (10 cm, e.g. LAD coronary)
R0             = 0.003  # Healthy (reference) lumen radius [m]  (3 mm)
STEN_SEVERITY  = 0.62   # Peak fractional radius reduction at stenosis  (0 → 1)
                        # 0.62 → ~62% diameter stenosis (clinically significant)
STEN_CENTRE    = 0.50   # Fractional axial position of the stenosis peak
STEN_SIGMA     = 0.012  # Gaussian half-width of the stenosis profile [m]

# ── Cardiac / flow parameters ─────────────────────────────────────────────────
HEART_RATE = 72              # [beats / min]  typical resting adult
OMEGA      = 2*np.pi * HEART_RATE / 60   # [rad / s]  angular cardiac frequency
Q_MEAN     = 1.3e-6          # Time-mean volumetric flow [m³/s]  (~1.3 mL/s)
                             # Physiological LAD coronary range: 0.5–3 mL/s
Q_AMP      = 0.75e-6         # Pulsatile amplitude [m³/s]
                             # Q(t) = Q_MEAN + Q_AMP·sin(ωt)

# ── Simulation / display parameters ──────────────────────────────────────────
N_SPATIAL   = 600   # Number of spatial grid points along artery
N_PARTICLES = 180   # Number of RBC proxy particles in the lumen
DT_SIM      = 0.007 # Simulation time-step [s]  (slightly accelerated for display)
FRAME_MS    = 30    # Milliseconds between animation frames  (~33 fps)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 2 ── BLOOD FLOW SIMULATION CLASS
# ═════════════════════════════════════════════════════════════════════════════

class BloodFlowSimulation:
    """
    1-D model of pulsatile blood flow through a Gaussian-shaped stenosis.

    Coordinate system
    -----------------
    x  : axial position along the artery  [m],  x ∈ [0, L]
    R  : local lumen radius               [m]
    A  : local cross-sectional area       [m²]  = π R²
    v  : axial blood velocity             [m/s]
    P  : gauge pressure (inlet = 0)       [Pa]
    Q  : volumetric flow rate             [m³/s]
    t  : simulation time                  [s]
    """

    def __init__(self):
        # ── Spatial grid ─────────────────────────────────────────────────────
        self.x  = np.linspace(0.0, L, N_SPATIAL)          # axial positions [m]
        self.dx = self.x[1] - self.x[0]                   # grid spacing [m]

        # ── Vessel geometry ───────────────────────────────────────────────────
        # The stenosis is modelled as a Gaussian reduction in lumen radius:
        #
        #   R(x) = R₀ · [1 − severity · exp(−(x − x_c)² / (2σ²))]
        #
        # This produces a smooth, symmetric narrowing centred at x_c,
        # physically representative of an eccentric plaque (atheroma).
        x_c    = STEN_CENTRE * L
        gauss  = np.exp(-((self.x - x_c)**2) / (2.0 * STEN_SIGMA**2))
        self.R = R0 * (1.0 - STEN_SEVERITY * gauss)       # radius profile [m]

        # Cross-sectional area A(x) = π R(x)²
        self.A  = np.pi * self.R**2                        # area profile [m²]

        # Reference quantities at the inlet (healthy segment)
        self.A0 = np.pi * R0**2                            # inlet area [m²]
        self.R_min = self.R.min()                          # throat radius [m]
        self.A_min = np.pi * self.R_min**2                 # throat area [m²]

        # ── Particles (proxy red blood cells) ─────────────────────────────────
        # Uniformly seed particles along x; random radial offset within lumen.
        self.px      = np.random.uniform(0.0, L, N_PARTICLES)   # x positions [m]
        # Normalised radial offset ∈ (−1, 1); will be scaled by local R each frame
        self.py_norm = np.random.uniform(-0.88, 0.88, N_PARTICLES)

        # ── Time ──────────────────────────────────────────────────────────────
        self.t = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    def flow_rate(self, t: float) -> float:
        """
        Pulsatile (sinusoidal) cardiac flow waveform.

        Q(t) = Q_mean + Q_amp · sin(ω t)

        A real Womersley solution would use Fourier harmonics of the aortic
        pressure waveform, but the single-harmonic model captures the
        systole/diastole alternation adequately for visualisation.
        """
        Q = Q_MEAN + Q_AMP * np.sin(OMEGA * t)
        return max(Q, 1e-10)   # guard against zero / negative (forward flow only)

    # ─────────────────────────────────────────────────────────────────────────
    def compute_fields(self, t: float):
        """
        Solve for velocity v(x) and pressure P(x) at time t.

        Returns
        -------
        v  : ndarray, shape (N_SPATIAL,) — velocity [m/s]
        P  : ndarray, shape (N_SPATIAL,) — gauge pressure [Pa]
        Q  : float  — instantaneous flow rate [m³/s]
        """
        Q = self.flow_rate(t)

        # ──────────────────────────────────────────────────────────────────────
        # VELOCITY  (from the continuity equation)
        # ──────────────────────────────────────────────────────────────────────
        # For incompressible, quasi-1-D flow:
        #
        #   A₁ v₁ = A₂ v₂ = Q  (everywhere along the artery)
        #
        #   →  v(x) = Q / A(x)
        #
        # Velocity is inversely proportional to area, so it spikes dramatically
        # where the lumen is narrowed (stenosis), consistent with clinical
        # Doppler ultrasound findings of elevated peak systolic velocity.
        # ──────────────────────────────────────────────────────────────────────
        v = Q / self.A   # velocity profile [m/s]

        # ──────────────────────────────────────────────────────────────────────
        # PRESSURE  (combined Bernoulli + Hagen-Poiseuille)
        # ──────────────────────────────────────────────────────────────────────
        # We superpose two contributions:
        #
        # (a) BERNOULLI (inviscid, reversible):
        #       P_B(x) = −½ρ [ v(x)² − v₀² ]
        #     Pressure falls at the stenosis entry (kinetic energy ↑) and
        #     *partially* recovers downstream (pressure recovery).
        #     The Bernoulli term alone would give full recovery; viscous
        #     dissipation prevents this in reality.
        #
        # (b) HAGEN-POISEUILLE GRADIENT (viscous, irreversible):
        #       dP_HP/dx = −8μQ / (π R(x)⁴)
        #     This is Poiseuille's law applied locally (valid when the flow
        #     is locally fully developed — a simplification here).
        #     Integrating along x gives a monotonically decreasing pressure.
        #
        # Total pressure: P(x) = P_B(x) + ∫₀ˣ dP_HP/dx' dx'
        # ──────────────────────────────────────────────────────────────────────
        v0 = Q / self.A0   # velocity at inlet (healthy lumen)

        # (a) Bernoulli component
        P_bernoulli = -0.5 * RHO * (v**2 - v0**2)

        # (b) Hagen-Poiseuille viscous pressure gradient
        dP_dx_hp = -8.0 * MU * Q / (np.pi * self.R**4)

        # Cumulative integral of viscous gradient (trapezoidal rule)
        P_hp = np.cumsum(dP_dx_hp) * self.dx
        P_hp -= P_hp[0]   # reference inlet to 0

        # Combined gauge pressure; re-reference inlet to 0
        P = P_bernoulli + P_hp
        P -= P[0]

        return v, P, Q

    # ─────────────────────────────────────────────────────────────────────────
    def advance_particles(self, v_field: np.ndarray):
        """
        Advect particle positions forward by one time-step DT_SIM.

        Each particle moves at the local flow velocity interpolated from
        the 1-D velocity field v(x).  Particles that exit the right
        boundary re-enter from the left with a fresh random radial offset,
        mimicking a continuous influx of red blood cells.

        Note: we use a simple forward-Euler scheme — sufficient for the
        visual fidelity required here.
        """
        # Interpolate velocity at each particle's x position
        v_p = np.interp(self.px, self.x, v_field)

        # Euler step
        self.px += v_p * DT_SIM

        # ── Wrap-around boundary condition ────────────────────────────────────
        exited = self.px >= L
        if exited.any():
            self.px[exited]      -= L
            self.py_norm[exited]  = np.random.uniform(-0.88, 0.88, exited.sum())

        # Handle rare negative positions (if Q momentarily negative — shouldn't
        # occur with our guard, but defensive coding)
        negative = self.px < 0.0
        if negative.any():
            self.px[negative] += L


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 3 ── VISUALISATION / ANIMATION CLASS
# ═════════════════════════════════════════════════════════════════════════════

# ── Colour palette (dark-theme "medical monitor" aesthetic) ───────────────────
C_BG        = '#080814'    # Figure background
C_AX        = '#0d0d22'    # Axis background
C_WALL      = '#cc3333'    # Artery wall colour
C_LUMEN     = '#1a0505'    # Lumen fill colour
C_GRID      = '#1e2240'    # Grid lines
C_TEXT      = '#b0b8d8'    # General text / tick labels
C_TITLE     = '#7a8fcc'    # Subplot titles
C_STEN      = '#ff7722'    # Stenosis marker
C_VEL_LINE  = '#ff3355'    # Velocity plot line
C_VEL_FILL  = '#ff3355'
C_PRES_LINE = '#33aaff'    # Pressure plot line
C_PRES_FILL = '#33aaff'
CMAP_RBC    = 'YlOrRd'     # Particle colour map  (yellow→red = slow→fast)


class HemodynamicAnimator:
    """
    Orchestrates figure layout, static artwork, and the FuncAnimation loop.

    Layout
    ------
    ┌─────────────────────────────────────────┐
    │          Artery animation               │  ← ax_artery  (full width)
    ├───────────────────┬─────────────────────┤
    │  Pressure profile │  Velocity profile   │  ← ax_pres, ax_vel
    └───────────────────┴─────────────────────┘
    """

    def __init__(self):
        self.sim = BloodFlowSimulation()
        self._build_figure()
        self._draw_static_artery()
        self._init_dynamic_artists()

    # ─────────────────────────────────────────────────────────────────────────
    def _build_figure(self):
        """Create figure, axes, and apply the dark theme."""
        plt.rcParams.update({
            'figure.facecolor':  C_BG,
            'text.color':        C_TEXT,
            'axes.labelcolor':   C_TEXT,
            'xtick.color':       C_TEXT,
            'ytick.color':       C_TEXT,
            'axes.edgecolor':    C_GRID,
            'grid.color':        C_GRID,
            'grid.linewidth':    0.5,
            'font.family':       'DejaVu Sans',
        })

        self.fig = plt.figure(figsize=(15, 8.5), facecolor=C_BG)
        self.fig.suptitle(
            'Hemodynamic Blood Flow Simulator  ·  Stenosed Artery (1-D Model)',
            color='white', fontsize=13, fontweight='bold', y=0.985
        )

        # Grid spec: 2 rows, 2 columns
        # Top row spans both columns (artery); bottom row split 50/50
        gs = self.fig.add_gridspec(
            2, 2,
            height_ratios=[1.9, 1.0],
            hspace=0.50,
            wspace=0.35,
            left=0.06, right=0.97,
            top=0.95, bottom=0.08,
        )

        self.ax_art  = self.fig.add_subplot(gs[0, :])   # top, full width
        self.ax_pres = self.fig.add_subplot(gs[1, 0])   # bottom-left
        self.ax_vel  = self.fig.add_subplot(gs[1, 1])   # bottom-right

        for ax in (self.ax_art, self.ax_pres, self.ax_vel):
            ax.set_facecolor(C_AX)
            ax.grid(True, alpha=0.35)

    # ─────────────────────────────────────────────────────────────────────────
    def _draw_static_artery(self):
        """
        Render artery walls and lumen fill on ax_art.
        These elements do not change between frames.
        """
        ax  = self.ax_art
        sim = self.sim
        xc  = sim.x * 1e2        # convert m → cm for display

        R_mm  =  sim.R * 1e3     # radius in mm (upper wall)
        R_mm_ = -sim.R * 1e3     # mirrored lower wall

        # ── Lumen fill ────────────────────────────────────────────────────────
        ax.fill_between(xc, R_mm_, R_mm,
                        color=C_LUMEN, alpha=0.90, zorder=1)

        # ── Wall boundary curves ──────────────────────────────────────────────
        ax.plot(xc, R_mm,  color=C_WALL, lw=2.2, solid_capstyle='round', zorder=5)
        ax.plot(xc, R_mm_, color=C_WALL, lw=2.2, solid_capstyle='round', zorder=5)

        # ── Stenosis marker ───────────────────────────────────────────────────
        xc_sten = STEN_CENTRE * L * 1e2
        ax.axvline(xc_sten, color=C_STEN, lw=1.0, ls='--', alpha=0.55, zorder=4)
        ax.text(xc_sten + 0.12,  R0*1e3*1.55,
                f'Stenosis\n({STEN_SEVERITY*100:.0f}% reduction)',
                color=C_STEN, fontsize=7.5, va='center', zorder=6)

        # ── Arrows: flow direction ────────────────────────────────────────────
        for xpos in [0.08, 0.85]:
            ax.annotate('', xy=(xpos + 0.05, 0), xytext=(xpos - 0.01, 0),
                        xycoords=('axes fraction', 'data'),
                        textcoords=('axes fraction', 'data'),
                        arrowprops=dict(arrowstyle='->', color='#446688',
                                        lw=1.2),
                        zorder=4)

        # ── Axis limits and labels ────────────────────────────────────────────
        self._art_ylim = R0 * 1e3 * 2.4   # half-height for display [mm]
        ax.set_xlim(0.0, L * 1e2)
        ax.set_ylim(-self._art_ylim, self._art_ylim)
        ax.set_xlabel('Axial position  [cm]', fontsize=9)
        ax.set_ylabel('Lumen radius  [mm]', fontsize=9)
        ax.set_title(
            'Arterial Lumen Cross-Section  (particle colour = relative velocity)',
            color=C_TITLE, fontsize=10, pad=6
        )

        # ── Metric annotations (right-hand side) ─────────────────────────────
        bbox_style = dict(facecolor='#0d0d22', edgecolor='#334466',
                          boxstyle='round,pad=0.35', alpha=0.85)
        self.text_t = ax.text(
            0.99, 0.93, '',
            transform=ax.transAxes,
            color='#ccddff', fontsize=8.5, ha='right', va='top',
            bbox=bbox_style, zorder=10
        )
        self.text_q = ax.text(
            0.99, 0.76, '',
            transform=ax.transAxes,
            color='#ccddff', fontsize=8.5, ha='right', va='top',
            bbox=bbox_style, zorder=10
        )

    # ─────────────────────────────────────────────────────────────────────────
    def _init_dynamic_artists(self):
        """
        Create the mutable Matplotlib artists (scatter, lines) that will be
        updated every animation frame.
        """
        sim = self.sim

        # ── Particle scatter ──────────────────────────────────────────────────
        # Use a single scatter object; offsets and colours updated per frame.
        norm_rbc = mcolors.Normalize(vmin=0.0, vmax=1.0)
        self.scat = self.ax_art.scatter(
            [], [],
            s=14, c=[], cmap=CMAP_RBC, norm=norm_rbc,
            alpha=0.85, edgecolors='none', zorder=7
        )

        # Colourbar (particle velocity key)
        sm = plt.cm.ScalarMappable(cmap=CMAP_RBC, norm=norm_rbc)
        sm.set_array([])
        cbar = self.fig.colorbar(sm, ax=self.ax_art,
                                 orientation='vertical',
                                 fraction=0.018, pad=0.01,
                                 shrink=0.85)
        cbar.set_label('Relative velocity', color=C_TEXT, fontsize=8)
        cbar.ax.tick_params(labelsize=7, colors=C_TEXT)
        cbar.outline.set_edgecolor('#334466')

        # ── Pressure subplot ──────────────────────────────────────────────────
        ax_p = self.ax_pres
        ax_p.set_title('Blood Pressure Profile', color=C_TITLE, fontsize=9, pad=5)
        ax_p.set_xlabel('Axial position  [cm]', fontsize=8)
        ax_p.set_ylabel('Gauge pressure  [Pa]', fontsize=8)
        ax_p.set_xlim(0.0, L * 1e2)
        # Y-limits: set dynamically each frame, but pre-initialise
        ax_p.set_ylim(-600.0, 50.0)
        ax_p.axhline(0.0, color='#334466', lw=0.7, ls=':')
        ax_p.axvline(STEN_CENTRE * L * 1e2,
                     color=C_STEN, lw=0.9, ls='--', alpha=0.5)
        ax_p.text(STEN_CENTRE * L * 1e2 + 0.12, -560,
                  'stenosis', color=C_STEN, fontsize=7, alpha=0.9)

        self.line_pres, = ax_p.plot([], [], color=C_PRES_LINE, lw=2.0, zorder=5)
        # Placeholder fill (removed and recreated each frame)
        self._fill_pres = None

        # ── Velocity subplot ──────────────────────────────────────────────────
        ax_v = self.ax_vel
        ax_v.set_title('Blood Velocity Profile', color=C_TITLE, fontsize=9, pad=5)
        ax_v.set_xlabel('Axial position  [cm]', fontsize=8)
        ax_v.set_ylabel('Velocity  [m/s]', fontsize=8)
        ax_v.set_xlim(0.0, L * 1e2)
        ax_v.set_ylim(0.0, 4.0)
        ax_v.axvline(STEN_CENTRE * L * 1e2,
                     color=C_STEN, lw=0.9, ls='--', alpha=0.5)
        ax_v.text(STEN_CENTRE * L * 1e2 + 0.12, 0.1,
                  'stenosis', color=C_STEN, fontsize=7, alpha=0.9)

        self.line_vel, = ax_v.plot([], [], color=C_VEL_LINE, lw=2.0, zorder=5)
        self._fill_vel = None

        # ── Phase indicator (cardiac cycle arc) ───────────────────────────────
        # Small text box showing systole / diastole phase
        self.text_phase_pres = ax_p.text(
            0.02, 0.93, '', transform=ax_p.transAxes,
            fontsize=7.5, color='#ffcc44',
            bbox=dict(facecolor='#0d0d22', edgecolor='none',
                      boxstyle='round,pad=0.3', alpha=0.8),
            zorder=10
        )
        self.text_phase_vel = ax_v.text(
            0.02, 0.93, '', transform=ax_v.transAxes,
            fontsize=7.5, color='#ffcc44',
            bbox=dict(facecolor='#0d0d22', edgecolor='none',
                      boxstyle='round,pad=0.3', alpha=0.8),
            zorder=10
        )

    # ─────────────────────────────────────────────────────────────────────────
    def _init_func(self):
        """FuncAnimation init callback — returns artists to blank."""
        self.scat.set_offsets(np.empty((0, 2)))
        self.line_pres.set_data([], [])
        self.line_vel.set_data([], [])
        self.text_t.set_text('')
        self.text_q.set_text('')
        self.text_phase_pres.set_text('')
        self.text_phase_vel.set_text('')
        return (self.scat, self.line_pres, self.line_vel,
                self.text_t, self.text_q)

    # ─────────────────────────────────────────────────────────────────────────
    def _update(self, frame: int):
        """
        FuncAnimation update callback — called once per frame.

        Steps
        -----
        1. Compute physics fields (v, P) for current time t.
        2. Advect particles in the velocity field.
        3. Update all dynamic Matplotlib artists.
        4. Increment simulation time.
        """
        sim = self.sim
        t   = sim.t

        # ── 1. Physics ────────────────────────────────────────────────────────
        v, P, Q = sim.compute_fields(t)

        # ── 2. Advance particles ──────────────────────────────────────────────
        sim.advance_particles(v)

        # ── 3a. Update particle scatter ───────────────────────────────────────
        # Convert positions to display units (cm, mm)
        xc_p = sim.px * 1e2                                      # [cm]
        R_at_p = np.interp(sim.px, sim.x, sim.R)                 # [m]
        yc_p = sim.py_norm * R_at_p * 1e3                        # [mm]

        # Colour particles by normalised local velocity (0 = slow, 1 = fast)
        v_at_p  = np.interp(sim.px, sim.x, v)
        v_min, v_max = v.min(), v.max()
        v_range = max(v_max - v_min, 1e-9)
        v_norm  = (v_at_p - v_min) / v_range

        self.scat.set_offsets(np.column_stack([xc_p, yc_p]))
        self.scat.set_array(v_norm)

        # ── 3b. Update pressure profile ───────────────────────────────────────
        xc = sim.x * 1e2   # [cm]
        self.line_pres.set_data(xc, P)

        # Replace fill_between (PolyCollection) cleanly
        if self._fill_pres is not None:
            self._fill_pres.remove()
        self._fill_pres = self.ax_pres.fill_between(
            xc, P, alpha=0.18, color=C_PRES_FILL, zorder=3
        )

        # Dynamic y-axis limits with 15% head-room
        p_lo = min(P.min() * 1.18, -10.0)
        p_hi = max(P.max() * 1.15,  10.0)
        self.ax_pres.set_ylim(p_lo, p_hi)

        # ── 3c. Update velocity profile ───────────────────────────────────────
        self.line_vel.set_data(xc, v)

        if self._fill_vel is not None:
            self._fill_vel.remove()
        self._fill_vel = self.ax_vel.fill_between(
            xc, v, alpha=0.18, color=C_VEL_FILL, zorder=3
        )

        self.ax_vel.set_ylim(0.0, v.max() * 1.22)

        # ── 3d. Info text ─────────────────────────────────────────────────────
        cycle_frac = (t % (2*np.pi / OMEGA)) / (2*np.pi / OMEGA)  # 0→1
        phase_label = 'SYSTOLE  ▲' if np.sin(OMEGA * t) > 0 else 'DIASTOLE ▼'
        phase_colour = '#ff6644' if np.sin(OMEGA * t) > 0 else '#44aaff'

        self.text_t.set_text(
            f'Time : {t:.3f} s\n'
            f'Cycle: {cycle_frac*100:.0f} %'
        )
        self.text_q.set_text(
            f'Q  = {Q*1e6:.2f} mL/s\n'
            f'v_max = {v.max():.2f} m/s\n'
            f'ΔP_total = {abs(P[-1]):.1f} Pa'
        )

        for txt in (self.text_phase_pres, self.text_phase_vel):
            txt.set_text(phase_label)
            txt.set_color(phase_colour)

        # ── 4. Advance time ───────────────────────────────────────────────────
        sim.t += DT_SIM

        # blit=False, so return value is less critical — but we return artists
        # for consistency (also allows easy switch to blit=True if fills removed)
        return (self.scat, self.line_pres, self.line_vel,
                self.text_t, self.text_q,
                self.text_phase_pres, self.text_phase_vel)

    # ─────────────────────────────────────────────────────────────────────────
    def run(self):
        """
        Launch the FuncAnimation and display the figure.

        blit=False is intentional: fill_between generates PolyCollections that
        are recreated each frame, which is incompatible with blit=True (blitting
        only refreshes known artists, not dynamically added collections).
        The performance cost is negligible for this spatial resolution.
        """
        self._anim = FuncAnimation(
            self.fig,
            self._update,
            init_func=self._init_func,
            frames=None,              # run indefinitely
            interval=FRAME_MS,        # ms between frames
            blit=False,
            cache_frame_data=False,   # prevent memory accumulation
        )

        # ── Summary panel (static, right of colourbar) ────────────────────────
        self.fig.text(
            0.50, 0.002,
            f'Blood: ρ={RHO} kg/m³  μ={MU} Pa·s  |  '
            f'Vessel: L={L*100:.0f} cm  R₀={R0*1000:.1f} mm  '
            f'Stenosis {STEN_SEVERITY*100:.0f}%  |  '
            f'HR={HEART_RATE} bpm',
            color='#556688', fontsize=7.5, ha='center', va='bottom'
        )

        plt.show()


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 4 ── ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 65)
    print("  Hemodynamic Blood Flow Simulator")
    print("  1-D Pulsatile Flow  ·  Gaussian Stenosis Model")
    print("=" * 65)
    print(f"  Blood density       ρ = {RHO} kg/m³")
    print(f"  Dynamic viscosity   μ = {MU} Pa·s  (non-Newtonian approx.)")
    print(f"  Artery length       L = {L*1e2:.0f} cm")
    print(f"  Normal radius      R₀ = {R0*1e3:.1f} mm")
    print(f"  Stenosis severity     = {STEN_SEVERITY*100:.0f}%  diameter reduction")
    print(f"  Heart rate            = {HEART_RATE} bpm  (ω = {OMEGA:.2f} rad/s)")
    print(f"  Mean flow rate     Q̄  = {Q_MEAN*1e6:.2f} mL/s")
    print(f"  Particles (RBCs)      = {N_PARTICLES}")
    print()

    # Compute and print a quick static summary at t = peak systole
    _sim = BloodFlowSimulation()
    _t_peak = 0.25 * (2 * np.pi / OMEGA)   # quarter-cycle = peak systole
    _v, _P, _Q = _sim.compute_fields(_t_peak)
    print(f"  Peak systole (t={_t_peak:.3f} s):")
    print(f"    Flow rate Q  = {_Q*1e6:.2f} mL/s")
    print(f"    v_inlet      = {_v[0]:.4f} m/s")
    print(f"    v_stenosis   = {_v.max():.4f} m/s  (×{_v.max()/_v[0]:.1f} acceleration)")
    print(f"    ΔP_total     = {abs(_P[-1]):.1f} Pa  ({abs(_P[-1])/133.322:.2f} mmHg)")
    print()
    print("  Close the figure window to exit.")
    print("=" * 65)

    animator = HemodynamicAnimator()
    animator.run()