#include "instrument_seno.h"
#include <math.h>
#include <iostream>

using namespace std;
using namespace upc;

InstrumentSeno::InstrumentSeno(const std::string &params)
: adsr(SamplingRate, params) {
   
    x.resize(BSIZE);
    bActive = false;
    
    KeyValue kv(params); 
    int N;
    if (!kv.to_int("N",N)) 
      N = 40;
    tbl.resize(N);
    float phase = 0;
    float stepTbl = 2 * 3.1415926 / (float) N;
    for (int i = 0; i < N; ++i) {
        tbl[i] = sin(phase);
        phase += stepTbl;
    }
    id = 0;
}


void InstrumentSeno::command(long cmd, long note, long vel) {
    if (cmd == 9) {    
        bActive = true;
        adsr.start();
        float freq = pow(2.0, (note - 69.0) / 12.0) * 440.0;
        step = freq * 2.0 * 3.1415926 / SamplingRate;
        A = vel / 127.0;
        phaseIndex = 0;
        id = 0;
    }
    else if (cmd == 8) { 
        adsr.stop();
    }
    else if (cmd == 0) { 
        adsr.end();
    }
}

const std::vector<float>& InstrumentSeno::synthesize() {
  if (!adsr.active()) {
      x.assign(x.size(), 0.0f);
      bActive = false;
      return x;
  }
  else if (!bActive)
    return x;
  for (unsigned int i = 0; i < x.size(); ++i) {
      phaseIndex += step;
      while (phaseIndex>2*3.1415926){
        phaseIndex -= 2*3.1415926;
      }
      const float TwoPi = 2.0f * 3.1415926f;
      id = (int)(phaseIndex * (float)tbl.size() / TwoPi);
      if (id >= (int)tbl.size()) id = 0;
      x[i] = A * tbl[id];
  }

  adsr(x);

  return x;
}