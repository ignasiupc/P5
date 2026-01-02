#pragma once

#include "instrument.h"
#include "envelope_adsr.h"
#include "keyvalue.h"

#include <vector>
#include <string>
#include <cmath>

namespace upc {

class InstrumentFM : public Instrument {
public:
    explicit InstrumentFM(const std::string& params);

    void command(long cmd, long note, long vel) override;
    const std::vector<float>& synthesize() override;

private:
    // Parámetros básicos del enunciado
    float N1 = 1.0f;     // ratio portadora respecto f0
    float N2 = 0.01f;    // ratio moduladora respecto f0
    float Isemi = 0.5f;  // I en semitonos (entrada)

    // Estado / audio
    std::vector<float> x;
    bool bActive = false;

    float A = 0.0f;      // amplitud (desde vel)
    double fc = 0.0;     // Hz
    double fm = 0.0;     // Hz
    double Iidx = 0.0;   // índice FM interno (adimensional)

    double phaseC = 0.0; // fase portadora (rad)
    double phaseM = 0.0; // fase moduladora (rad)
    double stepC  = 0.0; // incremento fase portadora por muestra (rad)
    double stepM  = 0.0; // incremento fase moduladora por muestra (rad)

    EnvelopeADSR adsr;

    // Helpers
    static double midi_to_hz(long note);
    static double semitones_to_index(double nu_semitones, double fc_hz, double fm_hz);
    static double wrap01(double phase);

    static inline double wrap2pi(double p) {
        const double TWO_PI = 6.283185307179586;
        while (p >= TWO_PI) p -= TWO_PI;
        while (p <  0.0)    p += TWO_PI;
        return p;
    }
};

} // namespace upc
