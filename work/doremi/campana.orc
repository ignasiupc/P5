# Campana (FM) — parámetros del artículo de Chowning (1973)
# Ejemplo bell-like: c/m = 200/280 = 1/1.4 => N1=1, N2=1.4
# Índice final ~10 en el artículo.
# Conversión a semitonos: I_st = 12*log2(1 + beta*(N2/N1)), beta=10 => I≈46.883 st

1 InstrumentFM ADSR_A=0.01; ADSR_D=1.20; ADSR_S=0.00; ADSR_R=0.35; N1=1; N2=1.4; I=46.883;
