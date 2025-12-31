
#pragma once
#include "instrument.h"
#include <vector>
#include "envelope_adsr.h"
#include "keyvalue.h"
namespace upc {
class InstrumentSeno : public Instrument {
    unsigned int id;
    EnvelopeADSR adsr;
    float A;                     
    std::vector<double> tbl;      
public:
    InstrumentSeno(const std::string &params);
    void command(long cmd, long note, long vel=1);
    const std::vector<float> & synthesize();
    float step;                 
    float phaseIndex;           
};
}
