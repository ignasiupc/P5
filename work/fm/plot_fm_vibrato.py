#!/usr/bin/env python3
"""Grafica un vibrato generado con InstrumentFM.

Produce una figura donde se ve:
- La señal temporal (zoom)
- La frecuencia instantánea estimada (Hilbert via FFT)
- La frecuencia instantánea teórica esperada a partir de N1, N2 e I

No requiere SciPy (solo numpy + matplotlib).
"""

from __future__ import annotations

import argparse
import math
import wave
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def midi_to_hz(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def read_wav_mono(path: str) -> tuple[int, np.ndarray]:
    """Lee WAV PCM (8/16/24/32) mono y devuelve (fs, x_float) en [-1,1]."""
    with wave.open(path, "rb") as w:
        nch = w.getnchannels()
        if nch != 1:
            raise ValueError(f"Se esperaba WAV mono, got channels={nch}")
        fs = w.getframerate()
        sampwidth = w.getsampwidth()
        nframes = w.getnframes()
        raw = w.readframes(nframes)

    if sampwidth == 1:
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        x = (x - 128.0) / 128.0
    elif sampwidth == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
        x = x / 32768.0
    elif sampwidth == 3:
        b = np.frombuffer(raw, dtype=np.uint8)
        a = b.reshape(-1, 3)
        x = (a[:, 0].astype(np.int32)
             | (a[:, 1].astype(np.int32) << 8)
             | (a[:, 2].astype(np.int32) << 16))
        x = np.where(x & 0x800000, x - 0x1000000, x).astype(np.float64)
        x = x / float(1 << 23)
    elif sampwidth == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float64)
        x = x / float(1 << 31)
    else:
        raise ValueError(f"sampwidth no soportado: {sampwidth}")

    return fs, x


def hilbert_fft(x: np.ndarray) -> np.ndarray:
    """Hilbert transform via FFT (analytical signal)."""
    N = x.size
    X = np.fft.fft(x)
    h = np.zeros(N, dtype=np.float64)
    if N % 2 == 0:
        h[0] = 1.0
        h[N // 2] = 1.0
        h[1:N // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(N + 1) // 2] = 2.0
    return np.fft.ifft(X * h)


def semitones_to_fd(Isemi: float, fc_hz: float) -> float:
    # coherente con InstrumentFM::semitones_to_index()
    return fc_hz * (2.0 ** (Isemi / 12.0) - 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", help="WAV mono generado por synth")
    ap.add_argument("--midi", type=int, default=69, help="Nota MIDI usada (default: 69=A4)")
    ap.add_argument("--N1", type=float, default=1.0, help="Múltiplo portador")
    ap.add_argument("--N2", type=float, default=0.013636, help="Múltiplo modulador (LFO)")
    ap.add_argument("--I", dest="Isemi", type=float, default=0.30, help="Índice en semitonos")
    ap.add_argument("--skip", type=float, default=0.05, help="Segundos a saltar (ataque)")
    ap.add_argument("--dur", type=float, default=2.5, help="Duración (s) a analizar")
    ap.add_argument("--zoom", type=float, default=0.25, help="Zoom temporal (s)")
    ap.add_argument("--out", default="work/fm/figs/fm_vibrato.png")
    args = ap.parse_args()

    fs, x = read_wav_mono(args.wav)

    i0 = int(max(0.0, args.skip) * fs)
    i1 = i0 + int(max(0.1, args.dur) * fs)
    i1 = min(i1, x.size)
    xw = x[i0:i1]

    # teoría a partir de N1,N2,I
    f0 = midi_to_hz(args.midi)
    fc = args.N1 * f0
    fm = args.N2 * f0
    fd = semitones_to_fd(args.Isemi, fc)

    # Estimación robusta de frecuencia instantánea:
    # 1) señal analítica (Hilbert)
    # 2) demodulación: z_bb = z * exp(-j 2π fc t) => fase baseband
    # 3) derivada de fase => desviación en Hz; sumamos fc
    z = hilbert_fft(xw)
    tt = np.arange(xw.size) / fs
    z_bb = z * np.exp(-1j * 2.0 * math.pi * fc * tt)
    phase_bb = np.unwrap(np.angle(z_bb))
    inst_dev = np.diff(phase_bb) * (fs / (2.0 * math.pi))
    inst_f = fc + inst_dev
    t = np.arange(inst_f.size) / fs

    # Suavizado (sin SciPy): media móvil corta para eliminar picos numéricos
    win = max(5, int(fs * 0.005))  # ~5 ms
    if win % 2 == 0:
        win += 1
    k = np.ones(win, dtype=np.float64) / float(win)
    inst_f_s = np.convolve(inst_f, k, mode="same")

    # Rechazo de outliers (saltos de fase puntuales): mantener alrededor de fc±3fd
    lo = fc - 3.0 * fd
    hi = fc + 3.0 * fd
    inst_f_s = np.where((inst_f_s >= lo) & (inst_f_s <= hi), inst_f_s, np.nan)

    f_theory = fc + fd * np.cos(2.0 * math.pi * fm * t)

    # plots (estilo simple y legible)
    for sty in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        try:
            plt.style.use(sty)
            break
        except Exception:
            pass

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(10.0, 6.0),
        gridspec_kw={"height_ratios": [1.0, 1.2]},
        constrained_layout=True,
    )

    # señal (zoom)
    tz = min(args.zoom, (xw.size / fs))
    iz = int(tz * fs)
    tt = np.arange(iz) / fs
    ax0.plot(tt * 1000.0, xw[:iz], lw=1.1)
    ax0.set_title("Señal FM (zoom)")
    ax0.set_xlabel("Tiempo (ms)")
    ax0.set_ylabel("Amplitud")
    ax0.grid(True, alpha=0.25)

    # frecuencia instantánea
    # (pelotitas para que se vea claro el muestreo)
    step = max(1, int(fs / 600.0))  # ~600 puntos/s
    ax1.plot(
        t[::step], inst_f_s[::step],
        ".", ms=2.2, alpha=0.55,
        label="f_inst estimada",
    )
    ax1.plot(
        t[::step], f_theory[::step],
        "-", lw=1.4, alpha=0.95,
        label="f_inst teórica",
    )
    ax1.set_title("Vibrato: f_inst estimada vs. teórica")
    ax1.set_xlabel("Tiempo (s)")
    ax1.set_ylabel("Frecuencia (Hz)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right")

    if fd > 0:
        ax1.set_ylim(fc - 1.4 * fd, fc + 1.4 * fd)

    txt = (
        f"N1={args.N1:g}  (fc={fc:.2f} Hz)\n"
        f"N2={args.N2:g}  (fm={fm:.2f} Hz)\n"
        f"I={args.Isemi:g} st  (fd={fd:.2f} Hz)"
    )
    ax1.text(
        0.02, 0.98, txt,
        transform=ax1.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Vibrato por FM (InstrumentFM)", fontsize=13)
    fig.savefig(out, dpi=220)
    print(f"[OK] Guardado: {out}")


if __name__ == "__main__":
    main()
