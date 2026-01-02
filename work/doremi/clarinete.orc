# Clarinete (FM) — parámetros del artículo de Chowning (1973)
# Ejemplo woodwind/clarinet-like: c/m = 900/600 = 3/2 => N1=3, N2=2
# En el artículo aparece índice entre 4 y 2; aquí usamos el valor 2 (estado estable).
# Conversión a semitonos: I_st = 12*log2(1 + beta*(N2/N1)), con beta=2 => I≈14.669 st

1 InstrumentFM ADSR_A=0.05; ADSR_D=0.10; ADSR_S=0.80; ADSR_R=0.15; N1=3; N2=2; I=14.669;
