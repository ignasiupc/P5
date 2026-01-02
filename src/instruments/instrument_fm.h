#pragma once
#include "instrument.h"
#include "envelope_adsr.h"
#include <string>

namespace upc {

class InstrumentFM : public Instrument {
public:
  explicit InstrumentFM(const std::string& params);

  // MIDI commands
  void command(long cmd, long note, long velocity) override;

  // Audio generation
  const std::vector<float>& synthesize() override;

private:
  // Params (from .orc)
  float N1 = 1.0f;     // carrier multiple
  float N2 = 1.0f;     // modulator multiple
  float Isemi = 0.0f;  // modulation index in semitones

  // State
  EnvelopeADSR adsr;

  double A = 0.0;      // amplitude (from velocity)
  double f0 = 0.0;     // note frequency
  double fc = 0.0;     // carrier frequency
  double fm = 0.0;     // modulator frequency
  double fd = 0.0;     // peak deviation
  double I = 0.0;      // classic FM index (fd/fm)

  double phc = 0.0;    // carrier phase
  double phm = 0.0;    // modulator phase
  double inc_c = 0.0;  // carrier phase increment
  double inc_m = 0.0;  // modulator phase increment

  static double midi_to_hz(long note);

  // Convert semitones -> classic index I = fd/fm using:
  // fd = fc * (2^(Isemi/12) - 1)
  static double semitones_to_index(double Isemi, double fc_hz, double fm_hz);
};

} // namespace upc
