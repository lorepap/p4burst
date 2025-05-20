// examples/custom_extern/extern.cpp

#include <chrono>
#include <cstdint>
#include <bm/bm_sim/extern.h>

using bm::Data;

class Time : public bm::ExternType {
 public:
  BM_EXTERN_ATTRIBUTES {};

  // init vuoto, richiesto dall’interfaccia
  void init() override {}

  // metodo esposto a P4: popola il Data& di output con il timestamp in ns
  void get_time_ns(Data &dst) {
    auto now = std::chrono::high_resolution_clock::now();
    uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()
    ).count();
    dst.set(ns);
  }
};

// registro l’extern e il suo metodo (con parametro Data&)
BM_REGISTER_EXTERN(Time);
BM_REGISTER_EXTERN_METHOD(Time, get_time_ns, Data &);
