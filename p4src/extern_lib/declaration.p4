// examples/custom_extern/declaration.p4

// Extern object per ottenere il tempo in nanosecondi.
extern Time {
  Time();
  // OUT parameter invece di return
  void get_time_ns(out bit<64> t);
}