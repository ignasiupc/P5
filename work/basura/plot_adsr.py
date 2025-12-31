import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

# Fitxers d'àudio i títols (els teus)
fitxers_audio = [
    ("work/generic_def.wav",    "Genèric"),
    ("work/perc_mant_def.wav",  "Percussiu mantingut"),
    ("work/perc_tall_def.wav",  "Percussiu tallat"),
    ("work/plana_def.wav",      "Pla")
]

plt.figure(figsize=(14, 10))

for i, (nom_fitxer, titol) in enumerate(fitxers_audio, 1):

    # --- ADSR segons els teus .orc ---
    if i == 1:  # Genèric
        A, D, S, R = 0.20, 0.30, 0.50, 0.40
    elif i == 2:  # Percussiu mantingut
        A, D, S, R = 0.005, 2.00, 0.00, 0.05
    elif i == 3:  # Percussiu tallat
        A, D, S, R = 0.005, 2.00, 0.00, 0.02
    elif i == 4:  # Pla
        A, D, S, R = 0.05, 0.08, 0.80, 0.07

    taxa_mostreig, dades = wav.read(nom_fitxer)

    # Si és estèreo, fem mitjana per passar-ho a mono
    if dades.ndim > 1:
        dades = dades.mean(axis=1)

    temps = np.arange(len(dades)) / taxa_mostreig

    # --- Detectar inici i final reals del so (per no posar 0.5 "a ull") ---
    absx = np.abs(dades.astype(np.float32))
    mx = absx.max() if absx.max() > 0 else 1.0
    thr = 0.02 * mx  # 2% del pic (puja-ho a 0.05 si tens clics)
    mask = absx > thr

    if np.any(mask):
        idx_on = np.argmax(mask)
        idx_off = len(mask) - 1 - np.argmax(mask[::-1])
        inici_nota = temps[idx_on]
        fi_nota = temps[idx_off]
    else:
        inici_nota = 0.0
        fi_nota = temps[-1]

    # --- Temps de fases ADSR (a partir del teu A D S R) ---
    atac_inici = inici_nota
    atac_fi = min(atac_inici + A, fi_nota)

    deca_inici = atac_fi
    deca_fi = min(deca_inici + D, fi_nota)

    # Sustained level (mateix criteri que els teus companys: amplitud en unitats del wav)
    nivell_sustain = S * mx

    release_fi = fi_nota
    release_inici = max(fi_nota - R, inici_nota)

    # Índexs per fases ADSR
    indices_atac = (temps >= atac_inici) & (temps <= atac_fi)
    indices_deca = (temps >= deca_inici) & (temps <= deca_fi)
    indices_release = (temps >= release_inici) & (temps <= release_fi)

    # Subgràfic
    plt.subplot(2, 2, i)
    plt.plot(temps, dades, label="Senyal d'àudio", alpha=0.7)

    # Mateixa manera de “pintar” que el codi dels teus companys
    plt.plot(temps[indices_atac], dades[indices_atac], color="red", label="Atac")
    plt.plot(temps[indices_deca], dades[indices_deca], color="blue", label="Decaiguda")
    plt.axhline(y=nivell_sustain, color="green", linestyle="--", label="Sosteniment")
    plt.plot(temps[indices_release], dades[indices_release], color="green", label="Alliberament")

    plt.title(titol)
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.savefig("work/adsr_grafiques_amics.png", dpi=200)
plt.show()
