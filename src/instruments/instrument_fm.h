#include "instrument_fm.h"
#include <algorithm>

static constexpr double TWOPI = 6.283185307179586476925286766559;

InstrumentFM::InstrumentFM(KeyValue& kv)
  : Instrument(kv), adsr(kv) // EnvelopeADSR lee ADSR_* del kv (igual que otros instrumentos)
{
  // N1, N2 e I (en semitonos) configurables desde el .orc
  kv.to_float("N1", N1);
  kv.to_float("N2", N2);
  kv.to_float("I",  Isemi);
}

double InstrumentFM::midi_to_hz(long note) {
  // f = 440 * 2^((note-69)/12)
  // (misma relación que se usa en el proyecto)
  return 440.0 * std::pow(2.0, (static_cast<double>(note) - 69.0) / 12.0);
}

double InstrumentFM::semitones_to_index(double nu_semitones, double fc_hz, double fm_hz) {
  // I = (fc/fm) * (2^(nu/12)-1)/(2^(nu/12)+1)
  // (nu en semitonos)
  // Nota: si fm=0, no hay modulación.
  if (fm_hz <= 1e-12) return 0.0;

  const double r = std::pow(2.0, nu_semitones / 12.0);
  const double num = (r - 1.0);
  const double den = (r + 1.0);
  return (fc_hz / fm_hz) * (num / den);
}

void InstrumentFM::command(long cmd, long note, long vel) {
  switch (cmd) {
    case 9: { // Note On
      const double f0 = midi_to_hz(note); // Hz
      fc = static_cast<double>(N1) * f0;
      fm = static_cast<double>(N2) * f0;

      // Convertimos I (semitonos) → índice FM interno
      Iidx = semitones_to_index(Isemi, fc, fm);

      // pasos en ciclos por muestra
      stepC = fc / static_cast<double>(SamplingRate);
      stepM = fm / static_cast<double>(SamplingRate);

      // amplitud desde velocity
      A = std::clamp(static_cast<float>(vel) / 127.0f, 0.0f, 1.0f);

      // reinicia fases y ADSR
      phaseC = 0.0;
      phaseM = 0.0;
      adsr.start();
    } break;

    case 8: // Note Off
      adsr.stop();
      break;

    default:
      break;
  }
}

void InstrumentFM::synthesize(std::vector<float>& out) {
  // Generamos FM y luego aplicamos ADSR sobre el buffer
  for (size_t i = 0; i < out.size(); ++i) {
    const double mod = std::sin(TWOPI * phaseM);
    const double arg = (TWOPI * phaseC) + (Iidx * mod);
    out[i] = static_cast<float>(A * std::sin(arg));

    phaseC = wrap01(phaseC + stepC);
    phaseM = wrap01(phaseM + stepM);
  }

  adsr(out);

  if (adsr.finished()) {
    // Si tu EnvelopeADSR tiene método finished/active, perfecto.
    // Si en tu versión se llama distinto, cámbialo por el equivalente.
    end();
  }
}

double InstrumentFM::wrap01(double phase) {
  return phase - std::floor(phase);
}