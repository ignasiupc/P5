import os
import re
import unicodedata
import numpy as np
import matplotlib.pyplot as plt


def safe_filename(text: str) -> str:
    """
    Convierte un texto a nombre de archivo seguro:
    - quita acentos
    - cambia espacios por _
    - elimina caracteres raros (incluye /)
    """
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = text.replace(" ", "_")
    # deja solo letras, numeros, _ y -
    text = re.sub(r"[^a-z0-9_-]+", "", text)
    # evita nombre vacío
    return text or "figura"


def adsr_curve(t, A, D, S, R, gate):
    """
    ADSR por tramos con NoteOff en 'gate' segundos.
    Si el NoteOff ocurre antes de acabar A o D, el release arranca del nivel actual.
    """
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)

    A = max(A, 0.0)
    D = max(D, 0.0)
    S = float(np.clip(S, 0.0, 1.0))
    R = max(R, 0.0)
    gate = max(gate, 0.0)

    def pre_release_level(tt):
        if tt <= 0:
            return 0.0
        if A > 0 and tt < A:
            return tt / A
        if D > 0 and tt < A + D:
            return 1.0 - (1.0 - S) * (tt - A) / D
        return S

    y_gate = pre_release_level(gate)

    for i, ti in enumerate(t):
        if ti < gate:
            y[i] = pre_release_level(ti)
        else:
            if R <= 0:
                y[i] = 0.0
            elif ti < gate + R:
                y[i] = y_gate * (1.0 - (ti - gate) / R)
            else:
                y[i] = 0.0

    return y


def plot_single(name, A, D, S, R, gate, total=None, outdir=None):
    # guarda figs al lado del script, para no depender del "pwd"
    if outdir is None:
        outdir = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(outdir, exist_ok=True)

    if total is None:
        total = max(gate + R, A + D) + 0.25

    t = np.linspace(0, total, 2000)
    y = adsr_curve(t, A, D, S, R, gate)

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=140)
    ax.plot(t, y, linewidth=2)

    xA = A
    xD = A + D
    xG = gate
    xR = gate + R

    for x, lab in [(xA, "A"), (xD, "A+D"), (xG, "NoteOff"), (xR, "Fin R")]:
        ax.axvline(x, linestyle="--", linewidth=1)
        ax.text(x, 1.02, lab, ha="center", va="bottom",
                transform=ax.get_xaxis_transform())

    # etiquetas de fases
    ax.text(min(xA/2, total*0.15), 0.55, "Ataque", va="center")
    ax.text(min((xA+xD)/2, total*0.35), max(S, 0.15), "Caida", va="center")
    ax.text(min((xD+xG)/2, total*0.55), S + 0.05 if S > 0 else 0.12, "Mantenimiento", va="center")
    ax.text(min((xG+xR)/2, total*0.80), 0.25, "Liberacion", va="center")

    ax.set_title(name)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud (normalizada)")
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, which="both", linestyle=":", linewidth=0.8)
    fig.tight_layout()

    outname = safe_filename(name) + ".png"
    outpath = os.path.join(outdir, outname)
    fig.savefig(outpath)
    plt.close(fig)
    print(f"[OK] Guardado: {outpath}")


def plot_overlay_percussive(A, D, S, R1, gate1, R2, gate2, outdir=None):
    if outdir is None:
        outdir = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(outdir, exist_ok=True)

    total = max(gate1 + R1, gate2 + R2, A + D) + 0.25
    t = np.linspace(0, total, 2000)

    y1 = adsr_curve(t, A, D, S, R1, gate1)
    y2 = adsr_curve(t, A, D, S, R2, gate2)

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=140)
    ax.plot(t, y1, linewidth=2, label="Nota mantenida (hasta extinguir)")
    ax.plot(t, y2, linewidth=2, label="Suelta temprana (corte)")

    ax.set_title("Percusivo — comparacion de finales de nota")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud (normalizada)")
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, which="both", linestyle=":", linewidth=0.8)
    ax.legend()
    fig.tight_layout()

    outpath = os.path.join(outdir, "percusivo_comparacion.png")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"[OK] Guardado: {outpath}")


if __name__ == "__main__":
    # mismos parámetros que en adsr_instruments.orc
    plot_single(
        name="ADSR generica",
        A=0.20, D=0.40, S=0.60, R=0.70,
        gate=1.40,
        total=2.40
    )

    plot_single(
        name="Percusivo nota mantenida hasta extinguir",
        A=0.005, D=1.50, S=0.00, R=0.20,    
        gate=2.00,
        total=2.40
    )

    plot_single(
        name="Percusivo suelta temprana con corte",
        A=0.005, D=1.50, S=0.00, R=0.02,
        gate=0.20,
        total=0.60
    )

    plot_single(
        name="Instrumento plano cuerdas frotadas o viento",
        A=0.05, D=0.02, S=0.85, R=0.08,
        gate=1.00,
        total=1.30
    )

    plot_overlay_percussive(
        A=0.005, D=1.50, S=0.00,
        R1=0.20, gate1=2.00,
        R2=0.02, gate2=0.20
    )
