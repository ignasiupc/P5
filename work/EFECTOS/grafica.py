#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

def read_wav_mono(path):
    """
    Lee wav mono. Intenta scipy, si no, usa wave (PCM int16 típico).
    Devuelve fs (Hz) y señal float en [-1,1].
    """
    try:
        from scipy.io import wavfile
        fs, x = wavfile.read(path)
        if x.ndim > 1:
            x = x[:, 0]
        # Normalización según dtype
        if np.issubdtype(x.dtype, np.integer):
            maxv = np.iinfo(x.dtype).max
            x = x.astype(np.float32) / maxv
        else:
            x = x.astype(np.float32)
        return fs, x
    except Exception:
        import wave
        with wave.open(path, "rb") as w:
            fs = w.getframerate()
            nchan = w.getnchannels()
            sampwidth = w.getsampwidth()
            nframes = w.getnframes()
            raw = w.readframes(nframes)

        if sampwidth != 2:
            raise RuntimeError("Solo se soporta PCM 16-bit si no está SciPy instalado.")
        x = np.frombuffer(raw, dtype=np.int16)
        if nchan > 1:
            x = x[::nchan]
        x = x.astype(np.float32) / 32768.0
        return fs, x

def moving_rms(x, win):
    win = max(3, int(win))
    w = np.ones(win, dtype=np.float32) / win
    return np.sqrt(np.convolve(x*x, w, mode="same") + 1e-12)

def envelope_hilbert_or_rms(x, fs, carrier_hint_hz=440.0):
    """
    Envolvente de amplitud:
      - si hay SciPy: Hilbert
      - si no: RMS móvil con ventana de unas pocas épocas del portador
    """
    try:
        from scipy.signal import hilbert
        env = np.abs(hilbert(x))
        return env
    except Exception:
        # ~5 periodos del portador como ventana
        win = int(fs * (5.0 / carrier_hint_hz))
        return moving_rms(x, win)

def estimate_inst_freq(x, fs):
    """
    Estima frecuencia instantánea.
    Preferible Hilbert (fase instantánea). Fallback: estimación por cruces por cero.
    """
    try:
        from scipy.signal import hilbert
        z = hilbert(x)
        phase = np.unwrap(np.angle(z))
        # dphi/dt / (2pi) -> Hz
        instf = np.diff(phase) * fs / (2*np.pi)
        return instf
    except Exception:
        # Cruces por cero: estimación grosera pero suficiente para “ver” vibrato
        s = np.sign(x)
        zc = np.where(np.diff(s) > 0)[0]  # cruces ascendentes
        t = zc / fs
        if len(t) < 3:
            return None
        periods = np.diff(t)
        f = 1.0 / periods
        # devolvemos una señal “escalonada” centrada en cada periodo
        instf = np.zeros_like(x, dtype=np.float32) * np.nan
        for k in range(len(periods)):
            i0, i1 = zc[k], zc[k+1]
            instf[i0:i1] = f[k]
        return instf

def annotate_period(ax, fm, y, x0=0.05):
    Tm = 1.0 / fm
    ax.annotate(
        "",
        xy=(x0 + Tm, y), xytext=(x0, y),
        arrowprops=dict(arrowstyle="<->", lw=1.5)
    )
    ax.text(x0 + Tm/2, y, f"  Tm = 1/fm = {Tm:.3f} s", va="bottom")

def plot_tremolo(wav_path, out_path, fm, A, window_s=0.8):
    fs, x = read_wav_mono(wav_path)
    N = int(min(len(x), window_s * fs))
    x = x[:N]
    t = np.arange(N) / fs

    env = envelope_hilbert_or_rms(x, fs)

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    ax.plot(t, x, linewidth=1.0, label="Señal (portadora)")
    ax.plot(t, env, linewidth=2.0, label="Envolvente |x(t)| (estimada)")
    ax.plot(t, -env, linewidth=2.0)

    ax.set_title("Trémolo sobre sinusoide: la amplitud oscila, la frecuencia del portador se mantiene")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud (normalizada)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # Info del efecto (explicación dentro de la figura)
    min_gain = 1 - 2*A
    max_gain = 1.0
    txt = (
        f"Parámetros del trémolo\n"
        f"fm = {fm:.2f} Hz  → velocidad de oscilación del volumen\n"
        f"A  = {A:.2f}     → profundidad (más A = más contraste)\n"
        f"Ganancia aprox. ∈ [{min_gain:.2f}, {max_gain:.2f}]"
    )
    ax.text(0.02, 0.02, txt, transform=ax.transAxes,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", alpha=0.85))

    annotate_period(ax, fm=fm, y=np.max(env)*0.85, x0=0.05)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

def plot_vibrato(wav_path, out_path, fm, I_semitones, f0=440.0, window_s=0.8):
    fs, x = read_wav_mono(wav_path)
    N = int(min(len(x), window_s * fs))
    x = x[:N]
    t = np.arange(N) / fs

    instf = estimate_inst_freq(x, fs)

    # límites teóricos a partir de I (semitonos)
    fmin = f0 * (2 ** (-I_semitones / 12.0))
    fmax = f0 * (2 ** ( I_semitones / 12.0))

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.15)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, x, linewidth=1.0)
    ax1.set_title("Vibrato sobre sinusoide: la amplitud se mantiene, cambia la frecuencia instantánea")
    ax1.set_ylabel("Amplitud (normalizada)")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    if instf is not None:
        tt = t[1:] if (instf is not None and len(instf) == len(t)-1) else t
        ax2.plot(tt, instf, linewidth=1.2, label="f_inst (estimada)")
    ax2.axhline(f0, linestyle="--", linewidth=1.2, label=f"f0 = {f0:.1f} Hz")
    ax2.axhline(fmin, linestyle=":", linewidth=1.2, label=f"fmin ≈ {fmin:.1f} Hz")
    ax2.axhline(fmax, linestyle=":", linewidth=1.2, label=f"fmax ≈ {fmax:.1f} Hz")
    ax2.set_xlabel("Tiempo (s)")
    ax2.set_ylabel("Hz")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    txt = (
        f"Parámetros del vibrato\n"
        f"fm = {fm:.2f} Hz  → rapidez del “temblor” de afinación\n"
        f"I  = {I_semitones:.2f} semitonos → extensión (más I = más desviación)\n"
        f"Rango teórico: [{fmin:.1f}, {fmax:.1f}] Hz alrededor de f0"
    )
    ax1.text(0.02, 0.05, txt, transform=ax1.transAxes,
             va="bottom", ha="left",
             bbox=dict(boxstyle="round,pad=0.4", alpha=0.85))

    annotate_period(ax2, fm=fm, y=fmax, x0=0.05)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tremolo_wav", default="tremolo.wav")
    ap.add_argument("--vibrato_wav", default="vibrato.wav")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--fm", type=float, default=6.0)
    ap.add_argument("--A", type=float, default=0.25)
    ap.add_argument("--I", type=float, default=0.50)
    ap.add_argument("--f0", type=float, default=440.0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    plot_tremolo(
        wav_path=args.tremolo_wav,
        out_path=os.path.join(args.outdir, "fig_tremolo.png"),
        fm=args.fm, A=args.A
    )
    plot_vibrato(
        wav_path=args.vibrato_wav,
        out_path=os.path.join(args.outdir, "fig_vibrato.png"),
        fm=args.fm, I_semitones=args.I, f0=args.f0
    )

    print("OK: generadas fig_tremolo.png y fig_vibrato.png")

if __name__ == "__main__":
    main()
