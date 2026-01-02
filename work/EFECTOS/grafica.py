#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import hilbert, butter, filtfilt

# -------------------------------------------------
# Utils
# -------------------------------------------------
def to_float_mono(x):
    """Stereo->mono and int->float approx in [-1,1]."""
    if x.ndim > 1:
        x = x.mean(axis=1)

    if np.issubdtype(x.dtype, np.integer):
        mx = np.iinfo(x.dtype).max
        x = x.astype(np.float32) / float(mx)
    else:
        x = x.astype(np.float32)
        m = float(np.max(np.abs(x)))
        if m > 0:
            x = x / m
    return x


def lowpass(y, fs, fc_hz=20.0, order=3):
    """Zero-phase lowpass."""
    fc = min(fc_hz, 0.45 * fs)
    b, a = butter(order, fc / (0.5 * fs), btype="low")
    return filtfilt(b, a, y)


def find_on_off(x, fs, thr_ratio=0.02, smooth_fc=10.0):
    """Detect active region by smoothed abs envelope."""
    env = lowpass(np.abs(x), fs, fc_hz=smooth_fc)
    m = float(np.max(env)) if np.max(env) > 0 else 1.0
    thr = thr_ratio * m
    mask = env > thr
    if not np.any(mask):
        return 0.0, (len(x) - 1) / fs, env
    i0 = int(np.argmax(mask))
    i1 = int(len(mask) - 1 - np.argmax(mask[::-1]))
    return i0 / fs, i1 / fs, env


def safe_filename(s):
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("_", "-", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def decimate_for_plot(t, y, max_points=150000):
    n = len(y)
    if n <= max_points:
        return t, y
    step = int(np.ceil(n / max_points))
    return t[::step], y[::step]


def zoom_window(t_start, t_end, fm_hz, n_cycles=6, min_s=0.8, max_s=3.0):
    """
    Return [tz0,tz1] so we visualize ~n_cycles of the modulation,
    clamped to [min_s, max_s] seconds.
    """
    if fm_hz <= 0:
        return t_start, t_end
    view = n_cycles / fm_hz
    view = max(min_s, min(max_s, view))
    mid = 0.5 * (t_start + t_end)
    tz0 = max(t_start, mid - view / 2)
    tz1 = min(t_end,   mid + view / 2)
    if tz1 <= tz0:
        return t_start, t_end
    return tz0, tz1


def inst_freq_from_hilbert(x, fs):
    """Instantaneous frequency from analytic signal."""
    z = hilbert(x)
    phi = np.unwrap(np.angle(z))
    dphi = np.diff(phi)
    finst = (fs / (2.0 * np.pi)) * dphi
    t = np.arange(len(finst)) / fs
    return t, finst


# -------------------------------------------------
# Plots
# -------------------------------------------------
def plot_tremolo(x, fs, outpath, fm_hz, A_index, title="TRÉMOLO"):
    t = np.arange(len(x)) / fs

    # active region
    t_on, t_off, _ = find_on_off(x, fs, thr_ratio=0.02, smooth_fc=10.0)

    # crop with small margins
    margin = 0.05
    t0 = max(0.0, t_on + margin)
    t1 = min(t[-1], t_off - margin)
    if t1 <= t0:
        t0, t1 = t_on, t_off

    idx = (t >= t0) & (t <= t1)
    tc = t[idx]
    xc = x[idx]
    if len(tc) < 10:
        tc = t
        xc = x

    # zoom to few cycles of modulation
    tz0, tz1 = zoom_window(tc[0], tc[-1], fm_hz, n_cycles=6, min_s=0.8, max_s=3.0)
    idz = (tc >= tz0) & (tc <= tz1)
    tc = tc[idz]
    xc = xc[idz]

    # envelope and normalization
    env = lowpass(np.abs(xc), fs, fc_hz=15.0)
    env_max = float(np.max(env)) if np.max(env) > 0 else 1.0
    env_min = float(np.min(env))
    envN = env / (env_max + 1e-12)  # normalized 0..1 approx

    # AM index estimate from envelope
    m_est = (np.max(envN) - np.min(envN)) / (np.max(envN) + np.min(envN) + 1e-12)

    # period annotation
    Tm = 1.0 / fm_hz if fm_hz > 0 else 0.0
    t_arrow0 = tc[0] + 0.25 * (tc[-1] - tc[0])
    t_arrow1 = t_arrow0 + Tm

    # Layout: plots left, info right
    fig = plt.figure(figsize=(13.5, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[3.2, 1.8])

    ax0 = fig.add_subplot(gs[0, 0])   # waveform
    ax1 = fig.add_subplot(gs[1, 0])   # envelope
    axI = fig.add_subplot(gs[:, 1])   # info box
    axI.axis("off")

    # waveform (zoomed)
    tt, xx = decimate_for_plot(tc, xc, max_points=160000)
    ax0.plot(tt, xx, lw=0.6, alpha=0.9, label="Señal (normalizada)")
    ax0.set_title(f"{title} — forma de onda (zoom)")
    ax0.set_ylabel("Amplitud")
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="upper right")

    # envelope normalized
    ax1.plot(tc, envN, lw=1.8)
    ax1.set_title("Envolvente |x(t)| normalizada (0..1) — aquí se ve el trémolo")
    ax1.set_xlabel("Tiempo (s)")
    ax1.set_ylabel("Envolvente normalizada")
    ax1.set_ylim(-0.05, 1.10)
    ax1.grid(True, alpha=0.25)

    # period arrow on envelope plot
    if fm_hz > 0 and t_arrow1 <= tc[-1]:
        y_arrow = 0.92
        ax1.annotate(
            "",
            xy=(t_arrow0, y_arrow),
            xytext=(t_arrow1, y_arrow),
            arrowprops=dict(arrowstyle="<->", lw=1.2),
        )
        ax1.text(
            0.5 * (t_arrow0 + t_arrow1),
            y_arrow,
            r"1 periodo  $T_m = 1/f_m$",
            ha="center",
            va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.9, ec="0.3"),
        )

    # Info (right side)
    eq = r"$y(t)=x(t)\,(1 + A\sin(2\pi f_m t))$"
    txt = (
        f"{eq}\n\n"
        f"Parámetros (definidos):\n"
        f"• f_m = {fm_hz:.2f} Hz  →  T_m = {Tm:.3f} s\n"
        f"• Índice A = {A_index:.2f}\n\n"
        f"Estimación (desde envolvente):\n"
        f"• A_est ≈ (max-min)/(max+min) = {m_est:.2f}\n\n"
        f"Ventana mostrada (zoom):\n"
        f"• {tc[0]:.2f} s  →  {tc[-1]:.2f} s  (Δt = {tc[-1]-tc[0]:.2f} s)\n"
    )
    axI.text(
        0.0, 1.0, txt,
        ha="left", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.6", fc="white", alpha=0.95, ec="0.3")
    )

    fig.suptitle("Figura — TRÉMOLO: f_m visible como periodicidad; A visible como profundidad", fontsize=14)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_vibrato(x, fs, outpath, fm_hz, I_semitones, title="VIBRATO"):
    t = np.arange(len(x)) / fs

    # active region
    t_on, t_off, _ = find_on_off(x, fs, thr_ratio=0.02, smooth_fc=10.0)

    margin = 0.05
    t0 = max(0.0, t_on + margin)
    t1 = min(t[-1], t_off - margin)
    if t1 <= t0:
        t0, t1 = t_on, t_off

    idx = (t >= t0) & (t <= t1)
    tc = t[idx]
    xc = x[idx]
    if len(tc) < 10:
        tc = t
        xc = x

    # zoom to few cycles
    tz0, tz1 = zoom_window(tc[0], tc[-1], fm_hz, n_cycles=6, min_s=0.8, max_s=3.0)
    idz = (tc >= tz0) & (tc <= tz1)
    tc = tc[idz]
    xc = xc[idz]

    # instantaneous frequency
    tf_rel, finst = inst_freq_from_hilbert(xc, fs)
    tf = tf_rel + tc[0]

    finst_s = lowpass(finst, fs, fc_hz=20.0)

    # trim edges a bit (avoid Hilbert border effects)
    trim_s = 0.02
    keep = (tf >= (tc[0] + trim_s)) & (tf <= (tc[-1] - trim_s))
    tf2 = tf[keep]
    finst2 = finst_s[keep]

    if len(finst2) > 0:
        f0_est = float(np.median(finst2))
        fmax = float(np.max(finst2))
        fmin = float(np.min(finst2))
        dF_est = 0.5 * (fmax - fmin)
    else:
        f0_est, dF_est = 0.0, 0.0

    # period arrow
    Tm = 1.0 / fm_hz if fm_hz > 0 else 0.0
    if len(tf2) > 0:
        t_arrow0 = tf2[0] + 0.25 * (tf2[-1] - tf2[0])
        t_arrow1 = t_arrow0 + Tm
    else:
        t_arrow0, t_arrow1 = 0.0, 0.0

    # Layout: plots left, info right
    fig = plt.figure(figsize=(13.5, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[3.2, 1.8])

    ax0 = fig.add_subplot(gs[0, 0])   # waveform
    ax1 = fig.add_subplot(gs[1, 0])   # f_inst
    axI = fig.add_subplot(gs[:, 1])   # info
    axI.axis("off")

    # waveform
    tt, xx = decimate_for_plot(tc, xc, max_points=160000)
    ax0.plot(tt, xx, lw=0.6, alpha=0.9, label="Señal (normalizada)")
    ax0.set_title(f"{title} — forma de onda (zoom)")
    ax0.set_ylabel("Amplitud")
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="upper right")

    # instantaneous frequency plot
    ax1.plot(tf2, finst2, lw=1.8)
    ax1.set_title(r"Frecuencia instantánea estimada $f_{inst}(t)$ — aquí se ve el vibrato")
    ax1.set_xlabel("Tiempo (s)")
    ax1.set_ylabel("Frecuencia (Hz)")
    ax1.grid(True, alpha=0.25)

    # reference lines + tight ylim around f0 ± Δf
    if len(finst2) > 0:
        ax1.axhline(f0_est, ls="--", lw=1.2)
        ax1.axhline(f0_est + dF_est, ls=":", lw=1.1)
        ax1.axhline(f0_est - dF_est, ls=":", lw=1.1)

        pad = max(2.0, 0.35 * dF_est)  # ensure not too tight
        ax1.set_ylim((f0_est - 1.2*dF_est - pad), (f0_est + 1.2*dF_est + pad))

    # period arrow on f_inst plot
    if fm_hz > 0 and len(tf2) > 0 and t_arrow1 <= tf2[-1]:
        y_arrow = f0_est + 0.9 * (dF_est if dF_est > 0 else 5.0)
        ax1.annotate(
            "",
            xy=(t_arrow0, y_arrow),
            xytext=(t_arrow1, y_arrow),
            arrowprops=dict(arrowstyle="<->", lw=1.2),
        )
        ax1.text(
            0.5 * (t_arrow0 + t_arrow1),
            y_arrow,
            r"1 periodo  $T_m = 1/f_m$",
            ha="center",
            va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.9, ec="0.3"),
        )

    # Info (right)
    txt = (
        "Vibrato: modulación en frecuencia\n\n"
        "Lectura directa en f_inst(t):\n"
        "• f_m = periodicidad (ciclos por segundo)\n"
        "• Índice I ↑  ⇒  Δf ↑ (más desviación)\n\n"
        f"Parámetros (definidos):\n"
        f"• f_m = {fm_hz:.2f} Hz  →  T_m = {Tm:.3f} s\n"
        f"• Índice I = {I_semitones:.2f} semitonos\n\n"
        f"Estimación (desde f_inst):\n"
        f"• f0 ≈ {f0_est:.1f} Hz\n"
        f"• Δf ≈ {dF_est:.1f} Hz (semi-amplitud)\n\n"
        f"Ventana mostrada (zoom):\n"
        f"• {tc[0]:.2f} s  →  {tc[-1]:.2f} s  (Δt = {tc[-1]-tc[0]:.2f} s)\n"
    )
    axI.text(
        0.0, 1.0, txt,
        ha="left", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.6", fc="white", alpha=0.95, ec="0.3")
    )

    fig.suptitle("Figura — VIBRATO: f_m visible como periodicidad; I visible como Δf", fontsize=14)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tremolo_wav", default="tremolo.wav")
    ap.add_argument("--vibrato_wav", default="vibrato.wav")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--fm_trem", type=float, default=5.0)
    ap.add_argument("--A_trem", type=float, default=0.25)
    ap.add_argument("--fm_vib", type=float, default=5.0)
    ap.add_argument("--I_vib", type=float, default=0.70)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    fs_t, xt = wavfile.read(args.tremolo_wav)
    xt = to_float_mono(xt)

    fs_v, xv = wavfile.read(args.vibrato_wav)
    xv = to_float_mono(xv)

    out_t = os.path.join(args.outdir, safe_filename("fig_tremolo.png"))
    out_v = os.path.join(args.outdir, safe_filename("fig_vibrato.png"))

    plot_tremolo(xt, fs_t, out_t, fm_hz=args.fm_trem, A_index=args.A_trem)
    print(f"[OK] Generada: {out_t}")

    plot_vibrato(xv, fs_v, out_v, fm_hz=args.fm_vib, I_semitones=args.I_vib)
    print(f"[OK] Generada: {out_v}")


if __name__ == "__main__":
    main()
