import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

# =========================
# CONFIG DEL TEU CAS
# =========================
fitxers_audio = [
    # (wav, titol, A, D, S, R)
    ("work/generic_def.wav",    "Genèric",      0.20, 0.30, 0.50, 0.40),
    ("work/perc_mant_def.wav",  "Percussiu 1",  0.005, 2.00, 0.00, 0.05),
    ("work/perc_tall_def.wav",  "Percussiu 2",  0.005, 2.00, 0.00, 0.02),
    ("work/plana_def.wav",      "Pla",          0.05, 0.08, 0.80, 0.07),
]

THRESH = 0.01      # 1% per detectar inici/final del so
SMOOTH_MS = 8      # suavitzat per detectar onset/offset

# =========================
# HELPERS
# =========================
def to_mono(x):
    return x.mean(axis=1) if x.ndim > 1 else x

def smooth_abs(x, fs, smooth_ms=8):
    ax = np.abs(x.astype(np.float32))
    win = max(1, int((smooth_ms/1000.0) * fs))
    k = np.ones(win, dtype=np.float32) / win
    return np.convolve(ax, k, mode="same")

def find_on_off(env, fs, thresh_ratio=0.01):
    m = env.max() if env.max() > 0 else 1.0
    mask = env > (thresh_ratio * m)
    if not np.any(mask):
        return 0.0, (len(env)-1)/fs
    i0 = int(np.argmax(mask))
    i1 = int(len(mask) - 1 - np.argmax(mask[::-1]))
    return i0/fs, i1/fs

# =========================
# PLOT
# =========================
plt.figure(figsize=(14, 9))

for i, (nom_fitxer, titol, A, D, S, R) in enumerate(fitxers_audio, 1):
    fs, dades = wav.read(nom_fitxer)
    dades = to_mono(dades)

    # temps
    temps = np.arange(len(dades)) / fs

    # envolvent per detectar onset/offset
    env = smooth_abs(dades, fs, SMOOTH_MS)
    t_on, t_off = find_on_off(env, fs, THRESH)

    # límits de fases teòrics (clamp per seguretat)
    atac_inici = t_on
    atac_fi    = min(t_on + A, t_off)

    deca_inici = atac_fi
    deca_fi    = min(deca_inici + D, t_off)

    release_fi = t_off
    release_inici = max(t_on, t_off - R)

    # sustain entre final decay i inici release (si existeix)
    sustain_inici = deca_fi
    sustain_fi    = max(sustain_inici, release_inici)

    # nivell sustain (en unitats d’amplitud del wav)
    peak = float(np.max(np.abs(dades))) if np.max(np.abs(dades)) > 0 else 1.0
    nivell_sustain = S * peak

    # màscares
    idx_atac    = (temps >= atac_inici) & (temps <= atac_fi)
    idx_deca    = (temps >= deca_inici) & (temps <= deca_fi)
    idx_release = (temps >= release_inici) & (temps <= release_fi)
    idx_sus     = (temps >= sustain_inici) & (temps <= sustain_fi)

    plt.subplot(2, 2, i)

    # senyal d’àudio (línia fina)
    plt.plot(temps, dades, label="Senyal d'àudio", alpha=0.35, linewidth=1.0)

    # FASES (estil “blocs” com el dels teus companys)
    if np.any(idx_atac):
        plt.fill_between(temps[idx_atac], dades[idx_atac], -dades[idx_atac],
                         color="red", alpha=0.85, label="Atac")
    if np.any(idx_deca):
        plt.fill_between(temps[idx_deca], dades[idx_deca], -dades[idx_deca],
                         color="blue", alpha=0.85, label="Decaiguda")
    if np.any(idx_sus) and S > 0:
        plt.fill_between(temps[idx_sus], nivell_sustain, -nivell_sustain,
                         color="#7FB3D5", alpha=0.65, label="Sosteniment")
    # línia discontínua del sustain (encara que sigui 0)
    plt.axhline(y=nivell_sustain, color="green", linestyle="--", label="Sosteniment")

    if np.any(idx_release):
        plt.fill_between(temps[idx_release], dades[idx_release], -dades[idx_release],
                         color="green", alpha=0.85, label="Alliberament")

    plt.title(titol)
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.legend(loc="best")

plt.tight_layout()
plt.savefig("work/adsr_grafiques.png", dpi=200)
plt.show()
