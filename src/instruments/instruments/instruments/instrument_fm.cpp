#include "instrument_fm.h"
#include "keyvalue.h"
#include <algorithm>
#include <cmath>

static constexpr double TWOPI = 6.283185307179586476925286766559;
static constexpr int SamplingRate = 44100;

namespace upc {

InstrumentFM::InstrumentFM(const std::string& params)
  : Instrument(), adsr(SamplingRate), N1(1.0), N2(1.0), Isemi(0.0)
{
  KeyValue kv(params);
  // N1, N2 e I (en semitonos) configurables desde el .orc
  kv.to_float("N1", N1);
  kv.to_float("N2", N2);
  kv.to_float("I",  Isemi);
}

double InstrumentFM::midi_to_hz(long note) {
  return 440.0 * std::pow(2.0, (static_cast<double>(note) - 69.0) / 12.0);
}

double InstrumentFM::semitones_to_index(double nu_semitones, double fc_hz, double fm_hz) {
  if (fm_hz <= 1e-12) return 0.0;

  const double r = std::pow(2.0, nu_semitones / 12.0);
  const double num = (r - 1.0);
  const double den = (r + 1.0);
  return (fc_hz / fm_hz) * (num / den);
}

double InstrumentFM::wrap01(double phase) {
  phase = phase - std::floor(phase);
  return phase;
}

void InstrumentFM::command(long cmd, long note, long vel) {
  switch (cmd) {
    case 9: { // Note On
      const double f0 = midi_to_hz(note);
      fc = N1 * f0;
      fm = N2 * f0;

      Iidx = semitones_to_index(Isemi, fc, fm);

      stepC = fc / static_cast<double>(SamplingRate);
      stepM = fm / static_cast<double>(SamplingRate);

      A = std::clamp(static_cast<float>(vel) / 127.0f, 0.0f, 1.0f);

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
  for (size_t i = 0; i < out.size(); ++i) {
    const double mod = std::sin(TWOPI * phaseM);
    const double arg = (TWOPI * phaseC) + (Iidx * mod);
    out[i] = static_cast<float>(A * std::sin(arg));

    phaseC = wrap01(phaseC + stepC);
    phaseM = wrap01(phaseM + stepM);
  }

  adsr(out);
}

}
