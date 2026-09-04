
#ifdef __AVX512__
template<>
constexpr int __micro_kernel_size<float, float, float>() {
    return 16;
}

void __micro_kernel_times_A_B(
    float* __restrict A, int ldA, float* __restrict B, int ldB,
    float* __restrict C, int ldC, int k)
{
    __m512 c0 = *(__m512*)(C);
    __m512 c1 = *(__m512*)(C + ldC);
    __m512 c2 = *(__m512*)(C + 2 * ldC);
    __m512 c3 = *(__m512*)(C + 3 * ldC);
    __m512 c4 = *(__m512*)(C + 4 * ldC);
    __m512 c5 = *(__m512*)(C + 5 * ldC);
    __m512 c6 = *(__m512*)(C + 6 * ldC);
    __m512 c7 = *(__m512*)(C + 7 * ldC);
    __m512 c8 = *(__m512*)(C + 8 * ldC);
    __m512 c9 = *(__m512*)(C + 9 * ldC);
    __m512 c10 = *(__m512*)(C + 10 * ldC);
    __m512 c11 = *(__m512*)(C + 11 * ldC);
    __m512 c12 = *(__m512*)(C + 12 * ldC);
    __m512 c13 = *(__m512*)(C + 13 * ldC);
    __m512 c14 = *(__m512*)(C + 14 * ldC);
    __m512 c15 = *(__m512*)(C + 15 * ldC);

    for (int p = 0; p < k; ++p) {
        __m512 a = *(__m512*)(A + p * ldA);
        float b0 = B[0 * ldB + p];
        float b1 = B[1 * ldB + p];
        float b2 = B[2 * ldB + p];
        float b3 = B[3 * ldB + p];
        float b4 = B[4 * ldB + p];
        float b5 = B[5 * ldB + p];
        float b6 = B[6 * ldB + p];
        float b7 = B[7 * ldB + p];
        float b8 = B[8 * ldB + p];
        float b9 = B[9 * ldB + p];
        float b10 = B[10 * ldB + p];
        float b11 = B[11 * ldB + p];
        float b12 = B[12 * ldB + p];
        float b13 = B[13 * ldB + p];
        float b14 = B[14 * ldB + p];
        float b15 = B[15 * ldB + p];

        c0 = _mm512_fmadd_ps(a, _mm512_set1_ps(b0), c0);
        c1 = _mm512_fmadd_ps(a, _mm512_set1_ps(b1), c1);
        c2 = _mm512_fmadd_ps(a, _mm512_set1_ps(b2), c2);
        c3 = _mm512_fmadd_ps(a, _mm512_set1_ps(b3), c3);
        c4 = _mm512_fmadd_ps(a, _mm512_set1_ps(b4), c4);
        c5 = _mm512_fmadd_ps(a, _mm512_set1_ps(b5), c5);
        c6 = _mm512_fmadd_ps(a, _mm512_set1_ps(b6), c6);
        c7 = _mm512_fmadd_ps(a, _mm512_set1_ps(b7), c7);
        c8 = _mm512_fmadd_ps(a, _mm512_set1_ps(b8), c8);
        c9 = _mm512_fmadd_ps(a, _mm512_set1_ps(b9), c9);
        c10 = _mm512_fmadd_ps(a, _mm512_set1_ps(b10), c10);
        c11 = _mm512_fmadd_ps(a, _mm512_set1_ps(b11), c11);
        c12 = _mm512_fmadd_ps(a, _mm512_set1_ps(b12), c12);
        c13 = _mm512_fmadd_ps(a, _mm512_set1_ps(b13), c13);
        c14 = _mm512_fmadd_ps(a, _mm512_set1_ps(b14), c14);
        c15 = _mm512_fmadd_ps(a, _mm512_set1_ps(b15), c15);
    }

    *(__m512*)(C) = c0;
    *(__m512*)(C + ldC) = c1;
    *(__m512*)(C + 2 * ldC) = c2;
    *(__m512*)(C + 3 * ldC) = c3;
    *(__m512*)(C + 4 * ldC) = c4;
    *(__m512*)(C + 5 * ldC) = c5;
    *(__m512*)(C + 6 * ldC) = c6;
    *(__m512*)(C + 7 * ldC) = c7;
    *(__m512*)(C + 8 * ldC) = c8;
    *(__m512*)(C + 9 * ldC) = c9;
    *(__m512*)(C + 10 * ldC) = c10;
    *(__m512*)(C + 11 * ldC) = c11;
    *(__m512*)(C + 12 * ldC) = c12;
    *(__m512*)(C + 13 * ldC) = c13;
    *(__m512*)(C + 14 * ldC) = c14;
    *(__m512*)(C + 15 * ldC) = c15;
}

// C -= A * B
void __micro_kernel_ntimes_A_B(
    float* __restrict A, int ldA, float* __restrict B, int ldB,
    float* __restrict C, int ldC, int k)
{
    __m512 c0 = *(__m512*)(C);
    __m512 c1 = *(__m512*)(C + ldC);
    __m512 c2 = *(__m512*)(C + 2 * ldC);
    __m512 c3 = *(__m512*)(C + 3 * ldC);
    __m512 c4 = *(__m512*)(C + 4 * ldC);
    __m512 c5 = *(__m512*)(C + 5 * ldC);
    __m512 c6 = *(__m512*)(C + 6 * ldC);
    __m512 c7 = *(__m512*)(C + 7 * ldC);
    __m512 c8 = *(__m512*)(C + 8 * ldC);
    __m512 c9 = *(__m512*)(C + 9 * ldC);
    __m512 c10 = *(__m512*)(C + 10 * ldC);
    __m512 c11 = *(__m512*)(C + 11 * ldC);
    __m512 c12 = *(__m512*)(C + 12 * ldC);
    __m512 c13 = *(__m512*)(C + 13 * ldC);
    __m512 c14 = *(__m512*)(C + 14 * ldC);
    __m512 c15 = *(__m512*)(C + 15 * ldC);

    for (int p = 0; p < k; ++p) {
        __m512 a = *(__m512*)(A + p * ldA);
        float b0 = B[0 * ldB + p];
        float b1 = B[1 * ldB + p];
        float b2 = B[2 * ldB + p];
        float b3 = B[3 * ldB + p];
        float b4 = B[4 * ldB + p];
        float b5 = B[5 * ldB + p];
        float b6 = B[6 * ldB + p];
        float b7 = B[7 * ldB + p];
        float b8 = B[8 * ldB + p];
        float b9 = B[9 * ldB + p];
        float b10 = B[10 * ldB + p];
        float b11 = B[11 * ldB + p];
        float b12 = B[12 * ldB + p];
        float b13 = B[13 * ldB + p];
        float b14 = B[14 * ldB + p];
        float b15 = B[15 * ldB + p];

        c0 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b0), c0);
        c1 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b1), c1);
        c2 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b2), c2);
        c3 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b3), c3);
        c4 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b4), c4);
        c5 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b5), c5);
        c6 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b6), c6);
        c7 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b7), c7);
        c8 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b8), c8);
        c9 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b9), c9);
        c10 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b10), c10);
        c11 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b11), c11);
        c12 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b12), c12);
        c13 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b13), c13);
        c14 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b14), c14);
        c15 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b15), c15);
    }

    *(__m512*)(C) = c0;
    *(__m512*)(C + ldC) = c1;
    *(__m512*)(C + 2 * ldC) = c2;
    *(__m512*)(C + 3 * ldC) = c3;
    *(__m512*)(C + 4 * ldC) = c4;
    *(__m512*)(C + 5 * ldC) = c5;
    *(__m512*)(C + 6 * ldC) = c6;
    *(__m512*)(C + 7 * ldC) = c7;
    *(__m512*)(C + 8 * ldC) = c8;
    *(__m512*)(C + 9 * ldC) = c9;
    *(__m512*)(C + 10 * ldC) = c10;
    *(__m512*)(C + 11 * ldC) = c11;
    *(__m512*)(C + 12 * ldC) = c12;
    *(__m512*)(C + 13 * ldC) = c13;
    *(__m512*)(C + 14 * ldC) = c14;
    *(__m512*)(C + 15 * ldC) = c15;
}

void __micro_kernel_ntimes_A_BT(
    float* __restrict A, int ldA, float* __restrict B, int ldB,
    float* __restrict C, int ldC, int k)
{
    __m512 c0 = *(__m512*)(C);
    __m512 c1 = *(__m512*)(C + ldC);
    __m512 c2 = *(__m512*)(C + 2 * ldC);
    __m512 c3 = *(__m512*)(C + 3 * ldC);
    __m512 c4 = *(__m512*)(C + 4 * ldC);
    __m512 c5 = *(__m512*)(C + 5 * ldC);
    __m512 c6 = *(__m512*)(C + 6 * ldC);
    __m512 c7 = *(__m512*)(C + 7 * ldC);
    __m512 c8 = *(__m512*)(C + 8 * ldC);
    __m512 c9 = *(__m512*)(C + 9 * ldC);
    __m512 c10 = *(__m512*)(C + 10 * ldC);
    __m512 c11 = *(__m512*)(C + 11 * ldC);
    __m512 c12 = *(__m512*)(C + 12 * ldC);
    __m512 c13 = *(__m512*)(C + 13 * ldC);
    __m512 c14 = *(__m512*)(C + 14 * ldC);
    __m512 c15 = *(__m512*)(C + 15 * ldC);

    for (int p = 0; p < k; ++p) {
        __m512 a = *(__m512*)(A + p * ldA);
        float b0 = B[0];
        float b1 = B[1];
        float b2 = B[2];
        float b3 = B[3];
        float b4 = B[4];
        float b5 = B[5];
        float b6 = B[6];
        float b7 = B[7];
        float b8 = B[8];
        float b9 = B[9];
        float b10 = B[10];
        float b11 = B[11];
        float b12 = B[12];
        float b13 = B[13];
        float b14 = B[14];
        float b15 = B[15];

        c0 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b0), c0);
        c1 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b1), c1);
        c2 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b2), c2);
        c3 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b3), c3);
        c4 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b4), c4);
        c5 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b5), c5);
        c6 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b6), c6);
        c7 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b7), c7);
        c8 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b8), c8);
        c9 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b9), c9);
        c10 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b10), c10);
        c11 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b11), c11);
        c12 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b12), c12);
        c13 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b13), c13);
        c14 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b14), c14);
        c15 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b15), c15);

        B += ldB;   // 移到下一列
    }

    *(__m512*)(C) = c0;
    *(__m512*)(C + ldC) = c1;
    *(__m512*)(C + 2 * ldC) = c2;
    *(__m512*)(C + 3 * ldC) = c3;
    *(__m512*)(C + 4 * ldC) = c4;
    *(__m512*)(C + 5 * ldC) = c5;
    *(__m512*)(C + 6 * ldC) = c6;
    *(__m512*)(C + 7 * ldC) = c7;
    *(__m512*)(C + 8 * ldC) = c8;
    *(__m512*)(C + 9 * ldC) = c9;
    *(__m512*)(C + 10 * ldC) = c10;
    *(__m512*)(C + 11 * ldC) = c11;
    *(__m512*)(C + 12 * ldC) = c12;
    *(__m512*)(C + 13 * ldC) = c13;
    *(__m512*)(C + 14 * ldC) = c14;
    *(__m512*)(C + 15 * ldC) = c15;
}

void __micro_kernel_ntimes_ADBT(
    float* __restrict A, int ldA, float* __restrict B, int ldB,
    float* __restrict C, int ldC, int k, float* d)
{
    __m512 c0 = *(__m512*)(C);
    __m512 c1 = *(__m512*)(C + ldC);
    __m512 c2 = *(__m512*)(C + 2 * ldC);
    __m512 c3 = *(__m512*)(C + 3 * ldC);
    __m512 c4 = *(__m512*)(C + 4 * ldC);
    __m512 c5 = *(__m512*)(C + 5 * ldC);
    __m512 c6 = *(__m512*)(C + 6 * ldC);
    __m512 c7 = *(__m512*)(C + 7 * ldC);
    __m512 c8 = *(__m512*)(C + 8 * ldC);
    __m512 c9 = *(__m512*)(C + 9 * ldC);
    __m512 c10 = *(__m512*)(C + 10 * ldC);
    __m512 c11 = *(__m512*)(C + 11 * ldC);
    __m512 c12 = *(__m512*)(C + 12 * ldC);
    __m512 c13 = *(__m512*)(C + 13 * ldC);
    __m512 c14 = *(__m512*)(C + 14 * ldC);
    __m512 c15 = *(__m512*)(C + 15 * ldC);

    for (int p = 0; p < k; ++p) {
        __m512 a = *(__m512*)(A + p * ldA);
        float b0 = B[0];
        float b1 = B[1];
        float b2 = B[2];
        float b3 = B[3];
        float b4 = B[4];
        float b5 = B[5];
        float b6 = B[6];
        float b7 = B[7];
        float b8 = B[8];
        float b9 = B[9];
        float b10 = B[10];
        float b11 = B[11];
        float b12 = B[12];
        float b13 = B[13];
        float b14 = B[14];
        float b15 = B[15];
        float D = d[p];

        c0 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b0 * D), c0);
        c1 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b1 * D), c1);
        c2 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b2 * D), c2);
        c3 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b3 * D), c3);
        c4 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b4 * D), c4);
        c5 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b5 * D), c5);
        c6 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b6 * D), c6);
        c7 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b7 * D), c7);
        c8 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b8 * D), c8);
        c9 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b9 * D), c9);
        c10 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b10 * D), c10);
        c11 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b11 * D), c11);
        c12 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b12 * D), c12);
        c13 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b13 * D), c13);
        c14 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b14 * D), c14);
        c15 = _mm512_fnmadd_ps(a, _mm512_set1_ps(b15 * D), c15);

        B += ldB;
    }

    *(__m512*)(C) = c0;
    *(__m512*)(C + ldC) = c1;
    *(__m512*)(C + 2 * ldC) = c2;
    *(__m512*)(C + 3 * ldC) = c3;
    *(__m512*)(C + 4 * ldC) = c4;
    *(__m512*)(C + 5 * ldC) = c5;
    *(__m512*)(C + 6 * ldC) = c6;
    *(__m512*)(C + 7 * ldC) = c7;
    *(__m512*)(C + 8 * ldC) = c8;
    *(__m512*)(C + 9 * ldC) = c9;
    *(__m512*)(C + 10 * ldC) = c10;
    *(__m512*)(C + 11 * ldC) = c11;
    *(__m512*)(C + 12 * ldC) = c12;
    *(__m512*)(C + 13 * ldC) = c13;
    *(__m512*)(C + 14 * ldC) = c14;
    *(__m512*)(C + 15 * ldC) = c15;
}


#elif defined(__AVX2__)
template<>
constexpr int __micro_kernel_size<float, float, float>() {
    return 8;
}

void __micro_kernel_times_A_B(
    float* __restrict A, int ldA, float* __restrict B, int ldB,
    float* __restrict C, int ldC, int k)
{
    __m256 c0 = *(__m256*)(C);
    __m256 c1 = *(__m256*)(C + ldC);
    __m256 c2 = *(__m256*)(C + 2 * ldC);
    __m256 c3 = *(__m256*)(C + 3 * ldC);
    __m256 c4 = *(__m256*)(C + 4 * ldC);
    __m256 c5 = *(__m256*)(C + 5 * ldC);
    __m256 c6 = *(__m256*)(C + 6 * ldC);
    __m256 c7 = *(__m256*)(C + 7 * ldC);

    for (int p = 0; p < k; ++p) {
        __m256 a = *(__m256*)(A + p * ldA);
        float b0 = B[0 * ldB + p];
        float b1 = B[1 * ldB + p];
        float b2 = B[2 * ldB + p];
        float b3 = B[3 * ldB + p];
        float b4 = B[4 * ldB + p];
        float b5 = B[5 * ldB + p];
        float b6 = B[6 * ldB + p];
        float b7 = B[7 * ldB + p];

        c0 = _mm256_fmadd_ps(a, _mm256_set1_ps(b0), c0);
        c1 = _mm256_fmadd_ps(a, _mm256_set1_ps(b1), c1);
        c2 = _mm256_fmadd_ps(a, _mm256_set1_ps(b2), c2);
        c3 = _mm256_fmadd_ps(a, _mm256_set1_ps(b3), c3);
        c4 = _mm256_fmadd_ps(a, _mm256_set1_ps(b4), c4);
        c5 = _mm256_fmadd_ps(a, _mm256_set1_ps(b5), c5);
        c6 = _mm256_fmadd_ps(a, _mm256_set1_ps(b6), c6);
        c7 = _mm256_fmadd_ps(a, _mm256_set1_ps(b7), c7);
    }

    *(__m256*)(C) = c0;
    *(__m256*)(C + ldC) = c1;
    *(__m256*)(C + 2 * ldC) = c2;
    *(__m256*)(C + 3 * ldC) = c3;
    *(__m256*)(C + 4 * ldC) = c4;
    *(__m256*)(C + 5 * ldC) = c5;
    *(__m256*)(C + 6 * ldC) = c6;
    *(__m256*)(C + 7 * ldC) = c7;
}

void __micro_kernel_ntimes_A_B(
    float* __restrict A, int ldA, float* __restrict B, int ldB,
    float* __restrict C, int ldC, int k)
{
    __m256 c0 = *(__m256*)(C);
    __m256 c1 = *(__m256*)(C + ldC);
    __m256 c2 = *(__m256*)(C + 2 * ldC);
    __m256 c3 = *(__m256*)(C + 3 * ldC);
    __m256 c4 = *(__m256*)(C + 4 * ldC);
    __m256 c5 = *(__m256*)(C + 5 * ldC);
    __m256 c6 = *(__m256*)(C + 6 * ldC);
    __m256 c7 = *(__m256*)(C + 7 * ldC);

    for (int p = 0; p < k; ++p) {
        __m256 a = *(__m256*)(A + p * ldA);
        float b0 = B[0 * ldB + p];
        float b1 = B[1 * ldB + p];
        float b2 = B[2 * ldB + p];
        float b3 = B[3 * ldB + p];
        float b4 = B[4 * ldB + p];
        float b5 = B[5 * ldB + p];
        float b6 = B[6 * ldB + p];
        float b7 = B[7 * ldB + p];

        c0 = _mm256_fnmadd_ps(a, _mm256_set1_ps(b0), c0);
        c1 = _mm256_fnmadd_ps(a, _mm256_set1_ps(b1), c1);
        c2 = _mm256_fnmadd_ps(a, _mm256_set1_ps(b2), c2);
        c3 = _mm256_fnmadd_ps(a, _mm256_set1_ps(b3), c3);
        c4 = _mm256_fnmadd_ps(a, _mm256_set1_ps(b4), c4);
        c5 = _mm256_fnmadd_ps(a, _mm256_set1_ps(b5), c5);
        c6 = _mm256_fnmadd_ps(a, _mm256_set1_ps(b6), c6);
        c7 = _mm256_fnmadd_ps(a, _mm256_set1_ps(b7), c7);
    }

    *(__m256*)(C) = c0;
    *(__m256*)(C + ldC) = c1;
    *(__m256*)(C + 2 * ldC) = c2;
    *(__m256*)(C + 3 * ldC) = c3;
    *(__m256*)(C + 4 * ldC) = c4;
    *(__m256*)(C + 5 * ldC) = c5;
    *(__m256*)(C + 6 * ldC) = c6;
    *(__m256*)(C + 7 * ldC) = c7;
}

void __micro_kernel_ntimes_A_BT(
    float* __restrict A, int ldA, float* __restrict B, int ldB,
    float* __restrict C, int ldC, int k)
{
    __m256 c0 = *(__m256*)(C);
    __m256 c1 = *(__m256*)(C + ldC);
    __m256 c2 = *(__m256*)(C + 2 * ldC);
    __m256 c3 = *(__m256*)(C + 3 * ldC);
    __m256 c4 = *(__m256*)(C + 4 * ldC);
    __m256 c5 = *(__m256*)(C + 5 * ldC);
    __m256 c6 = *(__m256*)(C + 6 * ldC);
    __m256 c7 = *(__m256*)(C + 7 * ldC);

    for (int p = 0; p < k; ++p) {
        __m256 a = *(__m256*)(A + p * ldA);
        float b0 = B[0];
        float b1 = B[1];
        float b2 = B[2];
        float b3 = B[3];
        float b4 = B[4];
        float b5 = B[5];
        float b6 = B[6];
        float b7 = B[7];

        c0 = _mm256_fnmadd_ps(a, _mm256_set1_ps(b0), c0);
        c1 = _mm256_fnmadd_ps(a, _mm256_set1_ps(b1), c1);
        c2 = _mm256_fnmadd_ps(a, _mm256_set1_ps(b2), c2);
        c3 = _mm256_fnmadd_ps(a, _mm256_set1_ps(b3), c3);
        c4 = _mm256_fnmadd_ps(a, _mm256_set1_ps(b4), c4);
        c5 = _mm256_fnmadd_ps(a, _mm256_set1_ps(b5), c5);
        c6 = _mm256_fnmadd_ps(a, _mm256_set1_ps(b6), c6);
        c7 = _mm256_fnmadd_ps(a, _mm256_set1_ps(b7), c7);

        B += ldB;
    }

    *(__m256*)(C) = c0;
    *(__m256*)(C + ldC) = c1;
    *(__m256*)(C + 2 * ldC) = c2;
    *(__m256*)(C + 3 * ldC) = c3;
    *(__m256*)(C + 4 * ldC) = c4;
    *(__m256*)(C + 5 * ldC) = c5;
    *(__m256*)(C + 6 * ldC) = c6;
    *(__m256*)(C + 7 * ldC) = c7;
}

void __micro_kernel_ntimes_ADBT(
    float* __restrict A, int ldA, float* __restrict B, int ldB,
    float* __restrict C, int ldC, int k, float* d)
{
    __m256 c0 = *(__m256*)(C);
    __m256 c1 = *(__m256*)(C + ldC);
    __m256 c2 = *(__m256*)(C + 2 * ldC);
    __m256 c3 = *(__m256*)(C + 3 * ldC);
    __m256 c4 = *(__m256*)(C + 4 * ldC);
    __m256 c5 = *(__m256*)(C + 5 * ldC);
    __m256 c6 = *(__m256*)(C + 6 * ldC);
    __m256 c7 = *(__m256*)(C + 7 * ldC);

    for (int p = 0; p < k; ++p) {
        __m256 a = *(__m256*)(A + p * ldA);
        float b0 = B[0];
        float b1 = B[1];
        float b2 = B[2];
        float b3 = B[3];
        float b4 = B[4];
        float b5 = B[5];
        float b6 = B[6];
        float b7 = B[7];
        float D = d[p];

        c0 = _mm256_fnmadd_ps(a, _mm256_set1_ps(b0 * D), c0);
        c1 = _mm256_fnmadd_ps(a, _mm256_set1_ps(b1 * D), c1);
        c2 = _mm256_fnmadd_ps(a, _mm256_set1_ps(b2 * D), c2);
        c3 = _mm256_fnmadd_ps(a, _mm256_set1_ps(b3 * D), c3);
        c4 = _mm256_fnmadd_ps(a, _mm256_set1_ps(b4 * D), c4);
        c5 = _mm256_fnmadd_ps(a, _mm256_set1_ps(b5 * D), c5);
        c6 = _mm256_fnmadd_ps(a, _mm256_set1_ps(b6 * D), c6);
        c7 = _mm256_fnmadd_ps(a, _mm256_set1_ps(b7 * D), c7);

        B += ldB;
    }

    *(__m256*)(C) = c0;
    *(__m256*)(C + ldC) = c1;
    *(__m256*)(C + 2 * ldC) = c2;
    *(__m256*)(C + 3 * ldC) = c3;
    *(__m256*)(C + 4 * ldC) = c4;
    *(__m256*)(C + 5 * ldC) = c5;
    *(__m256*)(C + 6 * ldC) = c6;
    *(__m256*)(C + 7 * ldC) = c7;
}

#endif