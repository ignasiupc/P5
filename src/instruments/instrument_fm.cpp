#include "instrument_fm.h"
#include "keyvalue.h"
#include <cmath>
#include <algorithm>

static constexpr double TWOPI = 6.283185307179586476925286766559;

namespace upc {

InstrumentFM::InstrumentFM(const std::string& params)
  : Instrument(), adsr(SamplingRate, params)
{
  KeyValue kv(params);
  kv.to_float("N1", N1);
  kv.to_float("N2", N2);
  kv.to_float("I",  Isemi);

  x.resize(BSIZE);
  bActive = false;
}

double InstrumentFM::midi_to_hz(long note) {
  return 440.0 * std::pow(2.0, (static_cast<double>(note) - 69.0) / 12.0);
}

double InstrumentFM::semitones_to_index(double Isemi, double fc_hz, double fm_hz) {
  // fd = fc*(2^(Isemi/12)-1)
  // I = fd/fm
  if (fm_hz <= 0.0) return 0.0;
  const double ratio = std::pow(2.0, Isemi / 12.0) - 1.0;
  const double fd = fc_hz * ratio;
  return fd / fm_hz;
}

void InstrumentFM::command(long cmd, long note, long velocity) {
  // Common: 9=note on, 8=note off, 0=end note
  if (cmd == 9) {
    f0 = midi_to_hz(note);

    // harmonic ratios:
    fc = static_cast<double>(N1) * f0;
    fm = static_cast<double>(N2) * f0;

    // semitones -> classic index
    I = semitones_to_index(Isemi, fc, fm);

    // phase increments
    inc_c = TWOPI * fc / SamplingRate;
    inc_m = TWOPI * fm / SamplingRate;

    phc = 0.0;
    phm = 0.0;

    // velocity -> amplitude (simple)
    A = static_cast<double>(velocity) / 127.0;
    A = std::max(0.0, std::min(1.0, A));

    adsr.start();
    bActive = true;
    return;
  }

  if (cmd == 8) {
    adsr.stop();
    return;
  }

  if (cmd == 0) {
    adsr.end();
    bActive = false;
    return;
  }
}

const std::vector<float>& InstrumentFM::synthesize() {
  if (!adsr.active()) {
    x.assign(x.size(), 0.0f);
    bActive = false;
    return x;
  }
  else if (!bActive) {
    return x;
  }

  for (unsigned int i = 0; i < x.size(); ++i) {
    const double y = A * std::sin(phc + I * std::sin(phm));
    x[i] = static_cast<float>(y);

    phc += inc_c;
    phm += inc_m;
    if (phc >= TWOPI) phc -= TWOPI;
    if (phm >= TWOPI) phm -= TWOPI;
  }

  adsr(x);
  return x;
}

} // namespace upc
