
#ifdef __AVX512__
template<>
constexpr int __micro_kernel_size<int, int, int>() {
    return 16;
}

void __micro_kernel_times_A_B(
    int* __restrict A, int ldA, int* __restrict B, int ldB,
    int* __restrict C, int ldC, int k)
{
    __m512i c0 = *(__m512i*)(C);
    __m512i c1 = *(__m512i*)(C + ldC);
    __m512i c2 = *(__m512i*)(C + 2 * ldC);
    __m512i c3 = *(__m512i*)(C + 3 * ldC);
    __m512i c4 = *(__m512i*)(C + 4 * ldC);
    __m512i c5 = *(__m512i*)(C + 5 * ldC);
    __m512i c6 = *(__m512i*)(C + 6 * ldC);
    __m512i c7 = *(__m512i*)(C + 7 * ldC);
    __m512i c8 = *(__m512i*)(C + 8 * ldC);
    __m512i c9 = *(__m512i*)(C + 9 * ldC);
    __m512i c10 = *(__m512i*)(C + 10 * ldC);
    __m512i c11 = *(__m512i*)(C + 11 * ldC);
    __m512i c12 = *(__m512i*)(C + 12 * ldC);
    __m512i c13 = *(__m512i*)(C + 13 * ldC);
    __m512i c14 = *(__m512i*)(C + 14 * ldC);
    __m512i c15 = *(__m512i*)(C + 15 * ldC);

    for (int p = 0; p < k; ++p) {
        __m512i a = *(__m512i*)(A + p * ldA);
        int b0 = B[0 * ldB + p];
        int b1 = B[1 * ldB + p];
        int b2 = B[2 * ldB + p];
        int b3 = B[3 * ldB + p];
        int b4 = B[4 * ldB + p];
        int b5 = B[5 * ldB + p];
        int b6 = B[6 * ldB + p];
        int b7 = B[7 * ldB + p];
        int b8 = B[8 * ldB + p];
        int b9 = B[9 * ldB + p];
        int b10 = B[10 * ldB + p];
        int b11 = B[11 * ldB + p];
        int b12 = B[12 * ldB + p];
        int b13 = B[13 * ldB + p];
        int b14 = B[14 * ldB + p];
        int b15 = B[15 * ldB + p];

        c0 = _mm512_add_epi32(c0, _mm512_mullo_epi32(a, _mm512_set1_epi32(b0)));
        c1 = _mm512_add_epi32(c1, _mm512_mullo_epi32(a, _mm512_set1_epi32(b1)));
        c2 = _mm512_add_epi32(c2, _mm512_mullo_epi32(a, _mm512_set1_epi32(b2)));
        c3 = _mm512_add_epi32(c3, _mm512_mullo_epi32(a, _mm512_set1_epi32(b3)));
        c4 = _mm512_add_epi32(c4, _mm512_mullo_epi32(a, _mm512_set1_epi32(b4)));
        c5 = _mm512_add_epi32(c5, _mm512_mullo_epi32(a, _mm512_set1_epi32(b5)));
        c6 = _mm512_add_epi32(c6, _mm512_mullo_epi32(a, _mm512_set1_epi32(b6)));
        c7 = _mm512_add_epi32(c7, _mm512_mullo_epi32(a, _mm512_set1_epi32(b7)));
        c8 = _mm512_add_epi32(c8, _mm512_mullo_epi32(a, _mm512_set1_epi32(b8)));
        c9 = _mm512_add_epi32(c9, _mm512_mullo_epi32(a, _mm512_set1_epi32(b9)));
        c10 = _mm512_add_epi32(c10, _mm512_mullo_epi32(a, _mm512_set1_epi32(b10)));
        c11 = _mm512_add_epi32(c11, _mm512_mullo_epi32(a, _mm512_set1_epi32(b11)));
        c12 = _mm512_add_epi32(c12, _mm512_mullo_epi32(a, _mm512_set1_epi32(b12)));
        c13 = _mm512_add_epi32(c13, _mm512_mullo_epi32(a, _mm512_set1_epi32(b13)));
        c14 = _mm512_add_epi32(c14, _mm512_mullo_epi32(a, _mm512_set1_epi32(b14)));
        c15 = _mm512_add_epi32(c15, _mm512_mullo_epi32(a, _mm512_set1_epi32(b15)));
    }

    *(__m512i*)(C) = c0;
    *(__m512i*)(C + ldC) = c1;
    *(__m512i*)(C + 2 * ldC) = c2;
    *(__m512i*)(C + 3 * ldC) = c3;
    *(__m512i*)(C + 4 * ldC) = c4;
    *(__m512i*)(C + 5 * ldC) = c5;
    *(__m512i*)(C + 6 * ldC) = c6;
    *(__m512i*)(C + 7 * ldC) = c7;
    *(__m512i*)(C + 8 * ldC) = c8;
    *(__m512i*)(C + 9 * ldC) = c9;
    *(__m512i*)(C + 10 * ldC) = c10;
    *(__m512i*)(C + 11 * ldC) = c11;
    *(__m512i*)(C + 12 * ldC) = c12;
    *(__m512i*)(C + 13 * ldC) = c13;
    *(__m512i*)(C + 14 * ldC) = c14;
    *(__m512i*)(C + 15 * ldC) = c15;
}

void __micro_kernel_ntimes_A_B(
    int* __restrict A, int ldA, int* __restrict B, int ldB,
    int* __restrict C, int ldC, int k)
{
    __m512i c0 = *(__m512i*)(C);
    __m512i c1 = *(__m512i*)(C + ldC);
    __m512i c2 = *(__m512i*)(C + 2 * ldC);
    __m512i c3 = *(__m512i*)(C + 3 * ldC);
    __m512i c4 = *(__m512i*)(C + 4 * ldC);
    __m512i c5 = *(__m512i*)(C + 5 * ldC);
    __m512i c6 = *(__m512i*)(C + 6 * ldC);
    __m512i c7 = *(__m512i*)(C + 7 * ldC);
    __m512i c8 = *(__m512i*)(C + 8 * ldC);
    __m512i c9 = *(__m512i*)(C + 9 * ldC);
    __m512i c10 = *(__m512i*)(C + 10 * ldC);
    __m512i c11 = *(__m512i*)(C + 11 * ldC);
    __m512i c12 = *(__m512i*)(C + 12 * ldC);
    __m512i c13 = *(__m512i*)(C + 13 * ldC);
    __m512i c14 = *(__m512i*)(C + 14 * ldC);
    __m512i c15 = *(__m512i*)(C + 15 * ldC);

    for (int p = 0; p < k; ++p) {
        __m512i a = *(__m512i*)(A + p * ldA);
        int b0 = B[0 * ldB + p];
        int b1 = B[1 * ldB + p];
        int b2 = B[2 * ldB + p];
        int b3 = B[3 * ldB + p];
        int b4 = B[4 * ldB + p];
        int b5 = B[5 * ldB + p];
        int b6 = B[6 * ldB + p];
        int b7 = B[7 * ldB + p];
        int b8 = B[8 * ldB + p];
        int b9 = B[9 * ldB + p];
        int b10 = B[10 * ldB + p];
        int b11 = B[11 * ldB + p];
        int b12 = B[12 * ldB + p];
        int b13 = B[13 * ldB + p];
        int b14 = B[14 * ldB + p];
        int b15 = B[15 * ldB + p];

        c0 = _mm512_sub_epi32(c0, _mm512_mullo_epi32(a, _mm512_set1_epi32(b0)));
        c1 = _mm512_sub_epi32(c1, _mm512_mullo_epi32(a, _mm512_set1_epi32(b1)));
        c2 = _mm512_sub_epi32(c2, _mm512_mullo_epi32(a, _mm512_set1_epi32(b2)));
        c3 = _mm512_sub_epi32(c3, _mm512_mullo_epi32(a, _mm512_set1_epi32(b3)));
        c4 = _mm512_sub_epi32(c4, _mm512_mullo_epi32(a, _mm512_set1_epi32(b4)));
        c5 = _mm512_sub_epi32(c5, _mm512_mullo_epi32(a, _mm512_set1_epi32(b5)));
        c6 = _mm512_sub_epi32(c6, _mm512_mullo_epi32(a, _mm512_set1_epi32(b6)));
        c7 = _mm512_sub_epi32(c7, _mm512_mullo_epi32(a, _mm512_set1_epi32(b7)));
        c8 = _mm512_sub_epi32(c8, _mm512_mullo_epi32(a, _mm512_set1_epi32(b8)));
        c9 = _mm512_sub_epi32(c9, _mm512_mullo_epi32(a, _mm512_set1_epi32(b9)));
        c10 = _mm512_sub_epi32(c10, _mm512_mullo_epi32(a, _mm512_set1_epi32(b10)));
        c11 = _mm512_sub_epi32(c11, _mm512_mullo_epi32(a, _mm512_set1_epi32(b11)));
        c12 = _mm512_sub_epi32(c12, _mm512_mullo_epi32(a, _mm512_set1_epi32(b12)));
        c13 = _mm512_sub_epi32(c13, _mm512_mullo_epi32(a, _mm512_set1_epi32(b13)));
        c14 = _mm512_sub_epi32(c14, _mm512_mullo_epi32(a, _mm512_set1_epi32(b14)));
        c15 = _mm512_sub_epi32(c15, _mm512_mullo_epi32(a, _mm512_set1_epi32(b15)));
    }

    *(__m512i*)(C) = c0;
    *(__m512i*)(C + ldC) = c1;
    *(__m512i*)(C + 2 * ldC) = c2;
    *(__m512i*)(C + 3 * ldC) = c3;
    *(__m512i*)(C + 4 * ldC) = c4;
    *(__m512i*)(C + 5 * ldC) = c5;
    *(__m512i*)(C + 6 * ldC) = c6;
    *(__m512i*)(C + 7 * ldC) = c7;
    *(__m512i*)(C + 8 * ldC) = c8;
    *(__m512i*)(C + 9 * ldC) = c9;
    *(__m512i*)(C + 10 * ldC) = c10;
    *(__m512i*)(C + 11 * ldC) = c11;
    *(__m512i*)(C + 12 * ldC) = c12;
    *(__m512i*)(C + 13 * ldC) = c13;
    *(__m512i*)(C + 14 * ldC) = c14;
    *(__m512i*)(C + 15 * ldC) = c15;
}

void __micro_kernel_ntimes_A_BT(
    int* __restrict A, int ldA, int* __restrict B, int ldB,
    int* __restrict C, int ldC, int k)
{
    __m512i c0 = *(__m512i*)(C);
    __m512i c1 = *(__m512i*)(C + ldC);
    __m512i c2 = *(__m512i*)(C + 2 * ldC);
    __m512i c3 = *(__m512i*)(C + 3 * ldC);
    __m512i c4 = *(__m512i*)(C + 4 * ldC);
    __m512i c5 = *(__m512i*)(C + 5 * ldC);
    __m512i c6 = *(__m512i*)(C + 6 * ldC);
    __m512i c7 = *(__m512i*)(C + 7 * ldC);
    __m512i c8 = *(__m512i*)(C + 8 * ldC);
    __m512i c9 = *(__m512i*)(C + 9 * ldC);
    __m512i c10 = *(__m512i*)(C + 10 * ldC);
    __m512i c11 = *(__m512i*)(C + 11 * ldC);
    __m512i c12 = *(__m512i*)(C + 12 * ldC);
    __m512i c13 = *(__m512i*)(C + 13 * ldC);
    __m512i c14 = *(__m512i*)(C + 14 * ldC);
    __m512i c15 = *(__m512i*)(C + 15 * ldC);

    for (int p = 0; p < k; ++p) {
        __m512i a = *(__m512i*)(A + p * ldA);
        int b0 = B[0];
        int b1 = B[1];
        int b2 = B[2];
        int b3 = B[3];
        int b4 = B[4];
        int b5 = B[5];
        int b6 = B[6];
        int b7 = B[7];
        int b8 = B[8];
        int b9 = B[9];
        int b10 = B[10];
        int b11 = B[11];
        int b12 = B[12];
        int b13 = B[13];
        int b14 = B[14];
        int b15 = B[15];

        c0 = _mm512_sub_epi32(c0, _mm512_mullo_epi32(a, _mm512_set1_epi32(b0)));
        c1 = _mm512_sub_epi32(c1, _mm512_mullo_epi32(a, _mm512_set1_epi32(b1)));
        c2 = _mm512_sub_epi32(c2, _mm512_mullo_epi32(a, _mm512_set1_epi32(b2)));
        c3 = _mm512_sub_epi32(c3, _mm512_mullo_epi32(a, _mm512_set1_epi32(b3)));
        c4 = _mm512_sub_epi32(c4, _mm512_mullo_epi32(a, _mm512_set1_epi32(b4)));
        c5 = _mm512_sub_epi32(c5, _mm512_mullo_epi32(a, _mm512_set1_epi32(b5)));
        c6 = _mm512_sub_epi32(c6, _mm512_mullo_epi32(a, _mm512_set1_epi32(b6)));
        c7 = _mm512_sub_epi32(c7, _mm512_mullo_epi32(a, _mm512_set1_epi32(b7)));
        c8 = _mm512_sub_epi32(c8, _mm512_mullo_epi32(a, _mm512_set1_epi32(b8)));
        c9 = _mm512_sub_epi32(c9, _mm512_mullo_epi32(a, _mm512_set1_epi32(b9)));
        c10 = _mm512_sub_epi32(c10, _mm512_mullo_epi32(a, _mm512_set1_epi32(b10)));
        c11 = _mm512_sub_epi32(c11, _mm512_mullo_epi32(a, _mm512_set1_epi32(b11)));
        c12 = _mm512_sub_epi32(c12, _mm512_mullo_epi32(a, _mm512_set1_epi32(b12)));
        c13 = _mm512_sub_epi32(c13, _mm512_mullo_epi32(a, _mm512_set1_epi32(b13)));
        c14 = _mm512_sub_epi32(c14, _mm512_mullo_epi32(a, _mm512_set1_epi32(b14)));
        c15 = _mm512_sub_epi32(c15, _mm512_mullo_epi32(a, _mm512_set1_epi32(b15)));

        B += ldB;
    }

    *(__m512i*)(C) = c0;
    *(__m512i*)(C + ldC) = c1;
    *(__m512i*)(C + 2 * ldC) = c2;
    *(__m512i*)(C + 3 * ldC) = c3;
    *(__m512i*)(C + 4 * ldC) = c4;
    *(__m512i*)(C + 5 * ldC) = c5;
    *(__m512i*)(C + 6 * ldC) = c6;
    *(__m512i*)(C + 7 * ldC) = c7;
    *(__m512i*)(C + 8 * ldC) = c8;
    *(__m512i*)(C + 9 * ldC) = c9;
    *(__m512i*)(C + 10 * ldC) = c10;
    *(__m512i*)(C + 11 * ldC) = c11;
    *(__m512i*)(C + 12 * ldC) = c12;
    *(__m512i*)(C + 13 * ldC) = c13;
    *(__m512i*)(C + 14 * ldC) = c14;
    *(__m512i*)(C + 15 * ldC) = c15;
}

void __micro_kernel_ntimes_ADBT(
    int* __restrict A, int ldA, int* __restrict B, int ldB,
    int* __restrict C, int ldC, int k, int* d)
{
    __m512i c0 = *(__m512i*)(C);
    __m512i c1 = *(__m512i*)(C + ldC);
    __m512i c2 = *(__m512i*)(C + 2 * ldC);
    __m512i c3 = *(__m512i*)(C + 3 * ldC);
    __m512i c4 = *(__m512i*)(C + 4 * ldC);
    __m512i c5 = *(__m512i*)(C + 5 * ldC);
    __m512i c6 = *(__m512i*)(C + 6 * ldC);
    __m512i c7 = *(__m512i*)(C + 7 * ldC);
    __m512i c8 = *(__m512i*)(C + 8 * ldC);
    __m512i c9 = *(__m512i*)(C + 9 * ldC);
    __m512i c10 = *(__m512i*)(C + 10 * ldC);
    __m512i c11 = *(__m512i*)(C + 11 * ldC);
    __m512i c12 = *(__m512i*)(C + 12 * ldC);
    __m512i c13 = *(__m512i*)(C + 13 * ldC);
    __m512i c14 = *(__m512i*)(C + 14 * ldC);
    __m512i c15 = *(__m512i*)(C + 15 * ldC);

    for (int p = 0; p < k; ++p) {
        __m512i a = *(__m512i*)(A + p * ldA);
        int b0 = B[0];
        int b1 = B[1];
        int b2 = B[2];
        int b3 = B[3];
        int b4 = B[4];
        int b5 = B[5];
        int b6 = B[6];
        int b7 = B[7];
        int b8 = B[8];
        int b9 = B[9];
        int b10 = B[10];
        int b11 = B[11];
        int b12 = B[12];
        int b13 = B[13];
        int b14 = B[14];
        int b15 = B[15];
        int D = d[p];

        c0 = _mm512_sub_epi32(c0, _mm512_mullo_epi32(a, _mm512_set1_epi32(b0 * D)));
        c1 = _mm512_sub_epi32(c1, _mm512_mullo_epi32(a, _mm512_set1_epi32(b1 * D)));
        c2 = _mm512_sub_epi32(c2, _mm512_mullo_epi32(a, _mm512_set1_epi32(b2 * D)));
        c3 = _mm512_sub_epi32(c3, _mm512_mullo_epi32(a, _mm512_set1_epi32(b3 * D)));
        c4 = _mm512_sub_epi32(c4, _mm512_mullo_epi32(a, _mm512_set1_epi32(b4 * D)));
        c5 = _mm512_sub_epi32(c5, _mm512_mullo_epi32(a, _mm512_set1_epi32(b5 * D)));
        c6 = _mm512_sub_epi32(c6, _mm512_mullo_epi32(a, _mm512_set1_epi32(b6 * D)));
        c7 = _mm512_sub_epi32(c7, _mm512_mullo_epi32(a, _mm512_set1_epi32(b7 * D)));
        c8 = _mm512_sub_epi32(c8, _mm512_mullo_epi32(a, _mm512_set1_epi32(b8 * D)));
        c9 = _mm512_sub_epi32(c9, _mm512_mullo_epi32(a, _mm512_set1_epi32(b9 * D)));
        c10 = _mm512_sub_epi32(c10, _mm512_mullo_epi32(a, _mm512_set1_epi32(b10 * D)));
        c11 = _mm512_sub_epi32(c11, _mm512_mullo_epi32(a, _mm512_set1_epi32(b11 * D)));
        c12 = _mm512_sub_epi32(c12, _mm512_mullo_epi32(a, _mm512_set1_epi32(b12 * D)));
        c13 = _mm512_sub_epi32(c13, _mm512_mullo_epi32(a, _mm512_set1_epi32(b13 * D)));
        c14 = _mm512_sub_epi32(c14, _mm512_mullo_epi32(a, _mm512_set1_epi32(b14 * D)));
        c15 = _mm512_sub_epi32(c15, _mm512_mullo_epi32(a, _mm512_set1_epi32(b15 * D)));

        B += ldB;
    }

    *(__m512i*)(C) = c0;
    *(__m512i*)(C + ldC) = c1;
    *(__m512i*)(C + 2 * ldC) = c2;
    *(__m512i*)(C + 3 * ldC) = c3;
    *(__m512i*)(C + 4 * ldC) = c4;
    *(__m512i*)(C + 5 * ldC) = c5;
    *(__m512i*)(C + 6 * ldC) = c6;
    *(__m512i*)(C + 7 * ldC) = c7;
    *(__m512i*)(C + 8 * ldC) = c8;
    *(__m512i*)(C + 9 * ldC) = c9;
    *(__m512i*)(C + 10 * ldC) = c10;
    *(__m512i*)(C + 11 * ldC) = c11;
    *(__m512i*)(C + 12 * ldC) = c12;
    *(__m512i*)(C + 13 * ldC) = c13;
    *(__m512i*)(C + 14 * ldC) = c14;
    *(__m512i*)(C + 15 * ldC) = c15;
}


#elif defined(__AVX2__)
template<>
constexpr int __micro_kernel_size<int, int, int>() {
    return 8;
}

void __micro_kernel_times_A_B(
    int* __restrict A, int ldA, int* __restrict B, int ldB,
    int* __restrict C, int ldC, int k)
{
    __m256i c0 = *(__m256i*)(C);
    __m256i c1 = *(__m256i*)(C + ldC);
    __m256i c2 = *(__m256i*)(C + 2 * ldC);
    __m256i c3 = *(__m256i*)(C + 3 * ldC);
    __m256i c4 = *(__m256i*)(C + 4 * ldC);
    __m256i c5 = *(__m256i*)(C + 5 * ldC);
    __m256i c6 = *(__m256i*)(C + 6 * ldC);
    __m256i c7 = *(__m256i*)(C + 7 * ldC);

    for (int p = 0; p < k; ++p) {
        __m256i a = *(__m256i*)(A + p * ldA);
        int b0 = B[0 * ldB + p];
        int b1 = B[1 * ldB + p];
        int b2 = B[2 * ldB + p];
        int b3 = B[3 * ldB + p];
        int b4 = B[4 * ldB + p];
        int b5 = B[5 * ldB + p];
        int b6 = B[6 * ldB + p];
        int b7 = B[7 * ldB + p];

        c0 = _mm256_add_epi32(c0, _mm256_mullo_epi32(a, _mm256_set1_epi32(b0)));
        c1 = _mm256_add_epi32(c1, _mm256_mullo_epi32(a, _mm256_set1_epi32(b1)));
        c2 = _mm256_add_epi32(c2, _mm256_mullo_epi32(a, _mm256_set1_epi32(b2)));
        c3 = _mm256_add_epi32(c3, _mm256_mullo_epi32(a, _mm256_set1_epi32(b3)));
        c4 = _mm256_add_epi32(c4, _mm256_mullo_epi32(a, _mm256_set1_epi32(b4)));
        c5 = _mm256_add_epi32(c5, _mm256_mullo_epi32(a, _mm256_set1_epi32(b5)));
        c6 = _mm256_add_epi32(c6, _mm256_mullo_epi32(a, _mm256_set1_epi32(b6)));
        c7 = _mm256_add_epi32(c7, _mm256_mullo_epi32(a, _mm256_set1_epi32(b7)));
    }

    *(__m256i*)(C) = c0;
    *(__m256i*)(C + ldC) = c1;
    *(__m256i*)(C + 2 * ldC) = c2;
    *(__m256i*)(C + 3 * ldC) = c3;
    *(__m256i*)(C + 4 * ldC) = c4;
    *(__m256i*)(C + 5 * ldC) = c5;
    *(__m256i*)(C + 6 * ldC) = c6;
    *(__m256i*)(C + 7 * ldC) = c7;
}

void __micro_kernel_ntimes_A_B(
    int* __restrict A, int ldA, int* __restrict B, int ldB,
    int* __restrict C, int ldC, int k)
{
    __m256i c0 = *(__m256i*)(C);
    __m256i c1 = *(__m256i*)(C + ldC);
    __m256i c2 = *(__m256i*)(C + 2 * ldC);
    __m256i c3 = *(__m256i*)(C + 3 * ldC);
    __m256i c4 = *(__m256i*)(C + 4 * ldC);
    __m256i c5 = *(__m256i*)(C + 5 * ldC);
    __m256i c6 = *(__m256i*)(C + 6 * ldC);
    __m256i c7 = *(__m256i*)(C + 7 * ldC);

    for (int p = 0; p < k; ++p) {
        __m256i a = *(__m256i*)(A + p * ldA);
        int b0 = B[0 * ldB + p];
        int b1 = B[1 * ldB + p];
        int b2 = B[2 * ldB + p];
        int b3 = B[3 * ldB + p];
        int b4 = B[4 * ldB + p];
        int b5 = B[5 * ldB + p];
        int b6 = B[6 * ldB + p];
        int b7 = B[7 * ldB + p];

        c0 = _mm256_sub_epi32(c0, _mm256_mullo_epi32(a, _mm256_set1_epi32(b0)));
        c1 = _mm256_sub_epi32(c1, _mm256_mullo_epi32(a, _mm256_set1_epi32(b1)));
        c2 = _mm256_sub_epi32(c2, _mm256_mullo_epi32(a, _mm256_set1_epi32(b2)));
        c3 = _mm256_sub_epi32(c3, _mm256_mullo_epi32(a, _mm256_set1_epi32(b3)));
        c4 = _mm256_sub_epi32(c4, _mm256_mullo_epi32(a, _mm256_set1_epi32(b4)));
        c5 = _mm256_sub_epi32(c5, _mm256_mullo_epi32(a, _mm256_set1_epi32(b5)));
        c6 = _mm256_sub_epi32(c6, _mm256_mullo_epi32(a, _mm256_set1_epi32(b6)));
        c7 = _mm256_sub_epi32(c7, _mm256_mullo_epi32(a, _mm256_set1_epi32(b7)));
    }

    *(__m256i*)(C) = c0;
    *(__m256i*)(C + ldC) = c1;
    *(__m256i*)(C + 2 * ldC) = c2;
    *(__m256i*)(C + 3 * ldC) = c3;
    *(__m256i*)(C + 4 * ldC) = c4;
    *(__m256i*)(C + 5 * ldC) = c5;
    *(__m256i*)(C + 6 * ldC) = c6;
    *(__m256i*)(C + 7 * ldC) = c7;
}

void __micro_kernel_ntimes_A_BT(
    int* __restrict A, int ldA, int* __restrict B, int ldB,
    int* __restrict C, int ldC, int k)
{
    __m256i c0 = *(__m256i*)(C);
    __m256i c1 = *(__m256i*)(C + ldC);
    __m256i c2 = *(__m256i*)(C + 2 * ldC);
    __m256i c3 = *(__m256i*)(C + 3 * ldC);
    __m256i c4 = *(__m256i*)(C + 4 * ldC);
    __m256i c5 = *(__m256i*)(C + 5 * ldC);
    __m256i c6 = *(__m256i*)(C + 6 * ldC);
    __m256i c7 = *(__m256i*)(C + 7 * ldC);

    for (int p = 0; p < k; ++p) {
        __m256i a = *(__m256i*)(A + p * ldA);
        int b0 = B[0];
        int b1 = B[1];
        int b2 = B[2];
        int b3 = B[3];
        int b4 = B[4];
        int b5 = B[5];
        int b6 = B[6];
        int b7 = B[7];

        c0 = _mm256_sub_epi32(c0, _mm256_mullo_epi32(a, _mm256_set1_epi32(b0)));
        c1 = _mm256_sub_epi32(c1, _mm256_mullo_epi32(a, _mm256_set1_epi32(b1)));
        c2 = _mm256_sub_epi32(c2, _mm256_mullo_epi32(a, _mm256_set1_epi32(b2)));
        c3 = _mm256_sub_epi32(c3, _mm256_mullo_epi32(a, _mm256_set1_epi32(b3)));
        c4 = _mm256_sub_epi32(c4, _mm256_mullo_epi32(a, _mm256_set1_epi32(b4)));
        c5 = _mm256_sub_epi32(c5, _mm256_mullo_epi32(a, _mm256_set1_epi32(b5)));
        c6 = _mm256_sub_epi32(c6, _mm256_mullo_epi32(a, _mm256_set1_epi32(b6)));
        c7 = _mm256_sub_epi32(c7, _mm256_mullo_epi32(a, _mm256_set1_epi32(b7)));

        B += ldB;
    }

    *(__m256i*)(C) = c0;
    *(__m256i*)(C + ldC) = c1;
    *(__m256i*)(C + 2 * ldC) = c2;
    *(__m256i*)(C + 3 * ldC) = c3;
    *(__m256i*)(C + 4 * ldC) = c4;
    *(__m256i*)(C + 5 * ldC) = c5;
    *(__m256i*)(C + 6 * ldC) = c6;
    *(__m256i*)(C + 7 * ldC) = c7;
}

void __micro_kernel_ntimes_ADBT(
    int* __restrict A, int ldA, int* __restrict B, int ldB,
    int* __restrict C, int ldC, int k, int* d)
{
    __m256i c0 = *(__m256i*)(C);
    __m256i c1 = *(__m256i*)(C + ldC);
    __m256i c2 = *(__m256i*)(C + 2 * ldC);
    __m256i c3 = *(__m256i*)(C + 3 * ldC);
    __m256i c4 = *(__m256i*)(C + 4 * ldC);
    __m256i c5 = *(__m256i*)(C + 5 * ldC);
    __m256i c6 = *(__m256i*)(C + 6 * ldC);
    __m256i c7 = *(__m256i*)(C + 7 * ldC);

    for (int p = 0; p < k; ++p) {
        __m256i a = *(__m256i*)(A + p * ldA);
        int b0 = B[0];
        int b1 = B[1];
        int b2 = B[2];
        int b3 = B[3];
        int b4 = B[4];
        int b5 = B[5];
        int b6 = B[6];
        int b7 = B[7];
        int D = d[p];

        c0 = _mm256_sub_epi32(c0, _mm256_mullo_epi32(a, _mm256_set1_epi32(b0 * D)));
        c1 = _mm256_sub_epi32(c1, _mm256_mullo_epi32(a, _mm256_set1_epi32(b1 * D)));
        c2 = _mm256_sub_epi32(c2, _mm256_mullo_epi32(a, _mm256_set1_epi32(b2 * D)));
        c3 = _mm256_sub_epi32(c3, _mm256_mullo_epi32(a, _mm256_set1_epi32(b3 * D)));
        c4 = _mm256_sub_epi32(c4, _mm256_mullo_epi32(a, _mm256_set1_epi32(b4 * D)));
        c5 = _mm256_sub_epi32(c5, _mm256_mullo_epi32(a, _mm256_set1_epi32(b5 * D)));
        c6 = _mm256_sub_epi32(c6, _mm256_mullo_epi32(a, _mm256_set1_epi32(b6 * D)));
        c7 = _mm256_sub_epi32(c7, _mm256_mullo_epi32(a, _mm256_set1_epi32(b7 * D)));

        B += ldB;
    }

    *(__m256i*)(C) = c0;
    *(__m256i*)(C + ldC) = c1;
    *(__m256i*)(C + 2 * ldC) = c2;
    *(__m256i*)(C + 3 * ldC) = c3;
    *(__m256i*)(C + 4 * ldC) = c4;
    *(__m256i*)(C + 5 * ldC) = c5;
    *(__m256i*)(C + 6 * ldC) = c6;
    *(__m256i*)(C + 7 * ldC) = c7;
}

#endif