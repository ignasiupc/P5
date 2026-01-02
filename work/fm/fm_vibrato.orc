# FM vibrato (usando InstrumentFM)
# Parámetros:
# - N1: múltiplo del portador (fc = N1*f0)
# - N2: múltiplo del modulador (fm = N2*f0)  -> usar N2 pequeño para LFO
# - I : índice en semitonos (profundidad de vibrato)

1 InstrumentFM ADSR_A=0.02; ADSR_D=0.10; ADSR_S=0.90; ADSR_R=0.15; N1=1; N2=0.013636; I=0.30;
