#include <cstdio>
#include "../src/cpu_matvec.cpp"
int main(){ std::printf("physical_cores=%d hardware_concurrency=%u\n", cpu::physical_cores(), std::thread::hardware_concurrency()); return 0; }
