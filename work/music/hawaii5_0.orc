# Orquestación (extra) para: Hawaii5-0
# Score original usa 16 canales; aquí se agrupan en familias para dar contraste.
# - Canales graves / bajo: timbre más estable y oscuro
# - Acompañamientos: timbre medio, poco brillo
# - Melodías / leads: timbre más brillante

# Bajo / graves (mucho material alrededor de MIDI 31–42)
1  InstrumentFM ADSR_A=0.01; ADSR_D=0.10; ADSR_S=0.70; ADSR_R=0.20; N1=1; N2=1; I=1.8;
2  InstrumentFM ADSR_A=0.01; ADSR_D=0.10; ADSR_S=0.70; ADSR_R=0.20; N1=1; N2=1; I=1.8;
3  InstrumentFM ADSR_A=0.01; ADSR_D=0.10; ADSR_S=0.70; ADSR_R=0.20; N1=1; N2=1; I=1.8;
4  InstrumentFM ADSR_A=0.01; ADSR_D=0.10; ADSR_S=0.70; ADSR_R=0.20; N1=1; N2=1; I=1.8;
5  InstrumentFM ADSR_A=0.01; ADSR_D=0.10; ADSR_S=0.70; ADSR_R=0.20; N1=1; N2=1; I=1.8;
6  InstrumentFM ADSR_A=0.01; ADSR_D=0.10; ADSR_S=0.70; ADSR_R=0.20; N1=1; N2=1; I=1.8;
8  InstrumentFM ADSR_A=0.01; ADSR_D=0.12; ADSR_S=0.65; ADSR_R=0.22; N1=1; N2=1; I=2.2;
14 InstrumentFM ADSR_A=0.01; ADSR_D=0.12; ADSR_S=0.65; ADSR_R=0.22; N1=1; N2=1; I=2.2;
15 InstrumentFM ADSR_A=0.01; ADSR_D=0.10; ADSR_S=0.70; ADSR_R=0.18; N1=1; N2=1; I=1.6;

# Acompañamientos / texturas medias
7  InstrumentFM ADSR_A=0.02; ADSR_D=0.18; ADSR_S=0.75; ADSR_R=0.20; N1=1; N2=2; I=3.0;
10 InstrumentFM ADSR_A=0.02; ADSR_D=0.18; ADSR_S=0.75; ADSR_R=0.20; N1=1; N2=2; I=3.0;
11 InstrumentFM ADSR_A=0.02; ADSR_D=0.18; ADSR_S=0.75; ADSR_R=0.22; N1=1; N2=2; I=3.2;
13 InstrumentFM ADSR_A=0.02; ADSR_D=0.18; ADSR_S=0.75; ADSR_R=0.22; N1=1; N2=2; I=3.2;

# Melodías / leads (rango alto: MIDI 72–98)
9  InstrumentFM ADSR_A=0.005; ADSR_D=0.12; ADSR_S=0.80; ADSR_R=0.14; N1=1; N2=3; I=5.5;
12 InstrumentFM ADSR_A=0.005; ADSR_D=0.12; ADSR_S=0.80; ADSR_R=0.14; N1=1; N2=3; I=5.5;
16 InstrumentFM ADSR_A=0.003; ADSR_D=0.10; ADSR_S=0.82; ADSR_R=0.12; N1=1; N2=4; I=6.0;
