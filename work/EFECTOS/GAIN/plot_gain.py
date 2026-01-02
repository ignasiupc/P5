#!/usr/bin/env python3
# plot_gain.py
# Compara 2 audios generados con distinta ganancia global (-g) en synth

import argparse
import wave
import numpy as np
import matplotlib.pyplot as plt


def read_wav_mono(path: str):
    """Lee WAV PCM y devuelve (fs, x_float_mono) en [-1, 1]."""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        fs = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Formato no soportado: sampwidth={sampwidth} bytes")

    if n_channels > 1:
        x = x.reshape(-1, n_channels).mean(axis=1)  # mono promedio

    return fs, x


def moving_rms(x: np.ndarray, win: int):
    """RMS móvil con ventana win muestras."""
    if win < 2:
        return np.abs(x)
    x2 = x * x
    kernel = np.ones(win, dtype=np.float32) / win
    m = np.convolve(x2, kernel, mode="same")
    return np.sqrt(m)


def stats(x: np.ndarray):
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x * x)))
    return peak, rms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav1", required=True, help="Primer WAV (ej. gain_g020.wav)")
    ap.add_argument("--wav2", required=True, help="Segundo WAV (ej. gain_g080.wav)")
    ap.add_argument("--label1", default="g1", help="Etiqueta 1 (ej. g=0.20)")
    ap.add_argument("--label2", default="g2", help="Etiqueta 2 (ej. g=0.80)")
    ap.add_argument("--out", default="gain_compare.png", help="Salida PNG")
    ap.add_argument("--zoom_ms", type=float, default=12.0, help="Zoom inicial (ms)")
    ap.add_argument("--env_ms", type=float, default=8.0, help="Ventana RMS (ms)")
    ap.add_argument("--t_view", type=float, default=0.35, help="Tiempo mostrado en panel largo (s)")
    args = ap.parse_args()

    fs1, x1 = read_wav_mono(args.wav1)
    fs2, x2 = read_wav_mono(args.wav2)
    if fs1 != fs2:
        raise ValueError("Los WAV deben tener el mismo samplerate")
    fs = fs1

    # Igualamos longitudes (por seguridad)
    N = min(len(x1), len(x2))
    x1 = x1[:N]
    x2 = x2[:N]
    t = np.arange(N) / fs

    # Estadísticas
    p1, r1 = stats(x1)
    p2, r2 = stats(x2)

    # Envolvente RMS para ver la amplitud sin “aplastar” la sinusoide
    win = int((args.env_ms / 1000.0) * fs)
    env1 = moving_rms(x1, win)
    env2 = moving_rms(x2, win)

    # Fig sin solapes
    fig = plt.figure(figsize=(11.5, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.62], height_ratios=[1, 1])

    ax_zoom = fig.add_subplot(gs[0, 0])
    ax_env  = fig.add_subplot(gs[1, 0])
    ax_txt  = fig.add_subplot(gs[:, 1])
    ax_txt.axis("off")

    # -------- Panel 1: Zoom
    t_zoom = args.zoom_ms / 1000.0
    i_zoom = int(t_zoom * fs)
    i_zoom = max(10, min(i_zoom, N))

    ax_zoom.plot(t[:i_zoom] * 1000.0, x1[:i_zoom], label=args.label1)
    ax_zoom.plot(t[:i_zoom] * 1000.0, x2[:i_zoom], label=args.label2)
    ax_zoom.set_title("Comparación en el dominio temporal (zoom inicial)")
    ax_zoom.set_xlabel("Tiempo (ms)")
    ax_zoom.set_ylabel("Amplitud")
    ax_zoom.grid(True, alpha=0.25)
    ax_zoom.legend(loc="upper right")

    # -------- Panel 2: Vista larga + envolvente RMS
    t_view = max(0.05, min(args.t_view, t[-1]))
    i_view = int(t_view * fs)

    ax_env.plot(t[:i_view], env1[:i_view], label=f"RMS móvil ({args.label1})")
    ax_env.plot(t[:i_view], env2[:i_view], label=f"RMS móvil ({args.label2})")
    ax_env.set_title("Evolución de amplitud (RMS móvil) — se aprecia el escalado por gain")
    ax_env.set_xlabel("Tiempo (s)")
    ax_env.set_ylabel("RMS (aprox.)")
    ax_env.grid(True, alpha=0.25)
    ax_env.legend(loc="upper right")

    # -------- Texto lateral (sin invadir la gráfica)
    txt = (
        "Efecto: GAIN global (volumen master)\n\n"
        "Qué cambia:\n"
        "• Solo la amplitud (x → g·x)\n"
        "• La frecuencia/tono NO cambia\n\n"
        "Cómo se ve en la señal:\n"
        "• La sinusoide mantiene el periodo\n"
        "• El valor pico y el RMS escalan con g\n\n"
        f"Medidas (WAV 1: {args.label1})\n"
        f"• Pico |x| = {p1:.3f}\n"
        f"• RMS     = {r1:.3f}\n\n"
        f"Medidas (WAV 2: {args.label2})\n"
        f"• Pico |x| = {p2:.3f}\n"
        f"• RMS     = {r2:.3f}\n\n"
        "Nota:\n"
        "• Si g es demasiado grande, puede haber clipping\n"
        "  (picos cerca de 1.0 en audio normalizado)."
    )
    ax_txt.text(0.0, 1.0, txt, va="top")

    fig.suptitle("Efecto Gain: comparación de dos señales sinusoidales con distinta ganancia", fontsize=13)
    fig.savefig(args.out, dpi=220)
    print(f"[OK] Guardado: {args.out}")


if __name__ == "__main__":
    main()
