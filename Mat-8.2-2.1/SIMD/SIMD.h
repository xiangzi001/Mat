
#ifndef XiangZiMatImport_SIMD
#define XiangZiMatImport_SIMD

#include<immintrin.h>

#ifdef __AVX512__
#include "__AVX512_double.h"
#include "__AVX512_float.h"
#include "__AVX512_int32.h"
#include "__AVX512_int64.h"

#elif defined(__AVX2__)
#include "__AVX2_double.h"
#include "__AVX2_float.h"
#include "__AVX2_int32.h"

#endif

template<typename t1, typename t2, typename t3>
constexpr int __micro_kernel_size() {
    return 4;
}

#include"__micro_kernel_double.h"
#include"__micro_kernel_float.h"
#include"__micro_kernel_int32.h"
#include"__micro_kernel_int64.h"

#endif // !XiangZiMatImport_SIMD