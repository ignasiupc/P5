#!/usr/bin/env python3
"""Comparación de vibrato por FM (varios WAV).

Genera figuras comparativas en las que se ve claramente cómo cambian:
- `I` (en semitonos): amplitud del vibrato (en semitonos)
- `N2`: velocidad del vibrato (Hz)

Estrategia:
- Estima f_inst de forma robusta (Hilbert + demodulación a fc + derivada de fase)
- Convierte a desviación en semitonos: delta_st = 12*log2(f_inst/fc)

Solo requiere numpy + matplotlib.
"""

from __future__ import annotations

import argparse
import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def midi_to_hz(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def read_wav_mono(path: str) -> tuple[int, np.ndarray]:
    with wave.open(path, "rb") as w:
        nch = w.getnchannels()
        if nch != 1:
            raise ValueError(f"Se esperaba WAV mono, got channels={nch}")
        fs = w.getframerate()
        sw = w.getsampwidth()
        n = w.getnframes()
        raw = w.readframes(n)

    if sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    elif sw == 1:
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        x = (x - 128.0) / 128.0
    else:
        raise ValueError(f"sampwidth no soportado para este script: {sw}")

    return fs, x


def hilbert_fft(x: np.ndarray) -> np.ndarray:
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
    return fc_hz * (2.0 ** (Isemi / 12.0) - 1.0)


def estimate_inst_freq(x: np.ndarray, fs: int, fc: float, fd: float) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (t, f_inst) estimados para x (mono)."""
    z = hilbert_fft(x)
    tt = np.arange(x.size) / fs
    z_bb = z * np.exp(-1j * 2.0 * math.pi * fc * tt)
    phase_bb = np.unwrap(np.angle(z_bb))
    inst_dev = np.diff(phase_bb) * (fs / (2.0 * math.pi))
    inst_f = fc + inst_dev
    t = np.arange(inst_f.size) / fs

    # suavizado y recorte razonable
    win = max(5, int(fs * 0.005))
    if win % 2 == 0:
        win += 1
    k = np.ones(win, dtype=np.float64) / float(win)
    inst_f = np.convolve(inst_f, k, mode="same")

    if fd > 0:
        lo = fc - 3.0 * fd
        hi = fc + 3.0 * fd
        inst_f = np.where((inst_f >= lo) & (inst_f <= hi), inst_f, np.nan)

    return t, inst_f


@dataclass(frozen=True)
class Case:
    wav: str
    label: str
    N1: float
    N2: float
    Isemi: float
    midi: int


def parse_case(s: str) -> Case:
    """Formato:
    wav:label:N1:N2:I:midi
    """
    parts = s.split(":")
    if len(parts) != 6:
        raise ValueError("--case debe tener formato wav:label:N1:N2:I:midi")
    wav, label, N1, N2, Isemi, midi = parts
    return Case(wav=wav, label=label, N1=float(N1), N2=float(N2), Isemi=float(Isemi), midi=int(midi))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", action="append", required=True, help="wav:label:N1:N2:I:midi (repetible)")
    ap.add_argument("--skip", type=float, default=0.06, help="Segundos a saltar (ataque)")
    ap.add_argument("--dur", type=float, default=1.7, help="Duración (s) a mostrar")
    ap.add_argument("--out", required=True, help="PNG de salida")
    args = ap.parse_args()

    cases = [parse_case(s) for s in args.case]

    # estilo sencillo
    for sty in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        try:
            plt.style.use(sty)
            break
        except Exception:
            pass

    fig, ax = plt.subplots(1, 1, figsize=(10.0, 4.8), constrained_layout=True)

    for c in cases:
        fs, x = read_wav_mono(c.wav)
        i0 = int(max(0.0, args.skip) * fs)
        i1 = i0 + int(max(0.2, args.dur) * fs)
        i1 = min(i1, x.size)
        xw = x[i0:i1]

        f0 = midi_to_hz(c.midi)
        fc = c.N1 * f0
        fm = c.N2 * f0
        fd = semitones_to_fd(c.Isemi, fc)

        t, f_inst = estimate_inst_freq(xw, fs, fc, fd)

        # desviación en semitonos alrededor de fc
        with np.errstate(divide="ignore", invalid="ignore"):
            delta_st = 12.0 * np.log2(f_inst / fc)

        # teoría (en semitonos, amplitud ~ I)
        delta_th = c.Isemi * np.cos(2.0 * math.pi * fm * t)

        step = max(1, int(fs / 700.0))
        ax.plot(t[::step], delta_st[::step], ".", ms=2.0, alpha=0.55, label=f"{c.label} (est)")
        ax.plot(t[::step], delta_th[::step], "-", lw=1.2, alpha=0.9, label=f"{c.label} (teo)")

    ax.set_title("Comparación de vibrato FM (desviación en semitonos)")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Desviación (semitonos)")
    ax.grid(True, alpha=0.25)
    ax.legend(ncols=2, fontsize=9, loc="upper right")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    print(f"[OK] Guardado: {out}")


if __name__ == "__main__":
    main()
