#ifdef __AVX512__

template<>
constexpr int __micro_kernel_size<int64_t, int64_t, int64_t>() {
    return 8;
}

void __micro_kernel_times_A_B(
    int64_t* __restrict A, int ldA, int64_t* __restrict B, int ldB,
    int64_t* __restrict C, int ldC, int k)
{
    __m512i c0 = *(__m512i*)(C);
    __m512i c1 = *(__m512i*)(C + ldC);
    __m512i c2 = *(__m512i*)(C + 2 * ldC);
    __m512i c3 = *(__m512i*)(C + 3 * ldC);
    __m512i c4 = *(__m512i*)(C + 4 * ldC);
    __m512i c5 = *(__m512i*)(C + 5 * ldC);
    __m512i c6 = *(__m512i*)(C + 6 * ldC);
    __m512i c7 = *(__m512i*)(C + 7 * ldC);

    for (int p = 0; p < k; ++p) {
        __m512i a = *(__m512i*)(A + p * ldA);
        int64_t b0 = B[0 * ldB + p];
        int64_t b1 = B[1 * ldB + p];
        int64_t b2 = B[2 * ldB + p];
        int64_t b3 = B[3 * ldB + p];
        int64_t b4 = B[4 * ldB + p];
        int64_t b5 = B[5 * ldB + p];
        int64_t b6 = B[6 * ldB + p];
        int64_t b7 = B[7 * ldB + p];

        c0 = _mm512_add_epi64(c0, _mm512_mullo_epi64(a, _mm512_set1_epi64(b0)));
        c1 = _mm512_add_epi64(c1, _mm512_mullo_epi64(a, _mm512_set1_epi64(b1)));
        c2 = _mm512_add_epi64(c2, _mm512_mullo_epi64(a, _mm512_set1_epi64(b2)));
        c3 = _mm512_add_epi64(c3, _mm512_mullo_epi64(a, _mm512_set1_epi64(b3)));
        c4 = _mm512_add_epi64(c4, _mm512_mullo_epi64(a, _mm512_set1_epi64(b4)));
        c5 = _mm512_add_epi64(c5, _mm512_mullo_epi64(a, _mm512_set1_epi64(b5)));
        c6 = _mm512_add_epi64(c6, _mm512_mullo_epi64(a, _mm512_set1_epi64(b6)));
        c7 = _mm512_add_epi64(c7, _mm512_mullo_epi64(a, _mm512_set1_epi64(b7)));
    }

    *(__m512i*)(C) = c0;
    *(__m512i*)(C + ldC) = c1;
    *(__m512i*)(C + 2 * ldC) = c2;
    *(__m512i*)(C + 3 * ldC) = c3;
    *(__m512i*)(C + 4 * ldC) = c4;
    *(__m512i*)(C + 5 * ldC) = c5;
    *(__m512i*)(C + 6 * ldC) = c6;
    *(__m512i*)(C + 7 * ldC) = c7;
}

void __micro_kernel_ntimes_A_B(
    int64_t* __restrict A, int ldA, int64_t* __restrict B, int ldB,
    int64_t* __restrict C, int ldC, int k)
{
    __m512i c0 = *(__m512i*)(C);
    __m512i c1 = *(__m512i*)(C + ldC);
    __m512i c2 = *(__m512i*)(C + 2 * ldC);
    __m512i c3 = *(__m512i*)(C + 3 * ldC);
    __m512i c4 = *(__m512i*)(C + 4 * ldC);
    __m512i c5 = *(__m512i*)(C + 5 * ldC);
    __m512i c6 = *(__m512i*)(C + 6 * ldC);
    __m512i c7 = *(__m512i*)(C + 7 * ldC);

    for (int p = 0; p < k; ++p) {
        __m512i a = *(__m512i*)(A + p * ldA);
        int64_t b0 = B[0 * ldB + p];
        int64_t b1 = B[1 * ldB + p];
        int64_t b2 = B[2 * ldB + p];
        int64_t b3 = B[3 * ldB + p];
        int64_t b4 = B[4 * ldB + p];
        int64_t b5 = B[5 * ldB + p];
        int64_t b6 = B[6 * ldB + p];
        int64_t b7 = B[7 * ldB + p];

        c0 = _mm512_sub_epi64(c0, _mm512_mullo_epi64(a, _mm512_set1_epi64(b0)));
        c1 = _mm512_sub_epi64(c1, _mm512_mullo_epi64(a, _mm512_set1_epi64(b1)));
        c2 = _mm512_sub_epi64(c2, _mm512_mullo_epi64(a, _mm512_set1_epi64(b2)));
        c3 = _mm512_sub_epi64(c3, _mm512_mullo_epi64(a, _mm512_set1_epi64(b3)));
        c4 = _mm512_sub_epi64(c4, _mm512_mullo_epi64(a, _mm512_set1_epi64(b4)));
        c5 = _mm512_sub_epi64(c5, _mm512_mullo_epi64(a, _mm512_set1_epi64(b5)));
        c6 = _mm512_sub_epi64(c6, _mm512_mullo_epi64(a, _mm512_set1_epi64(b6)));
        c7 = _mm512_sub_epi64(c7, _mm512_mullo_epi64(a, _mm512_set1_epi64(b7)));
    }

    *(__m512i*)(C) = c0;
    *(__m512i*)(C + ldC) = c1;
    *(__m512i*)(C + 2 * ldC) = c2;
    *(__m512i*)(C + 3 * ldC) = c3;
    *(__m512i*)(C + 4 * ldC) = c4;
    *(__m512i*)(C + 5 * ldC) = c5;
    *(__m512i*)(C + 6 * ldC) = c6;
    *(__m512i*)(C + 7 * ldC) = c7;
}

void __micro_kernel_ntimes_A_BT(
    int64_t* __restrict A, int ldA, int64_t* __restrict B, int ldB,
    int64_t* __restrict C, int ldC, int k)
{
    __m512i c0 = *(__m512i*)(C);
    __m512i c1 = *(__m512i*)(C + ldC);
    __m512i c2 = *(__m512i*)(C + 2 * ldC);
    __m512i c3 = *(__m512i*)(C + 3 * ldC);
    __m512i c4 = *(__m512i*)(C + 4 * ldC);
    __m512i c5 = *(__m512i*)(C + 5 * ldC);
    __m512i c6 = *(__m512i*)(C + 6 * ldC);
    __m512i c7 = *(__m512i*)(C + 7 * ldC);

    for (int p = 0; p < k; ++p) {
        __m512i a = *(__m512i*)(A + p * ldA);
        int64_t b0 = B[0];
        int64_t b1 = B[1];
        int64_t b2 = B[2];
        int64_t b3 = B[3];
        int64_t b4 = B[4];
        int64_t b5 = B[5];
        int64_t b6 = B[6];
        int64_t b7 = B[7];

        c0 = _mm512_sub_epi64(c0, _mm512_mullo_epi64(a, _mm512_set1_epi64(b0)));
        c1 = _mm512_sub_epi64(c1, _mm512_mullo_epi64(a, _mm512_set1_epi64(b1)));
        c2 = _mm512_sub_epi64(c2, _mm512_mullo_epi64(a, _mm512_set1_epi64(b2)));
        c3 = _mm512_sub_epi64(c3, _mm512_mullo_epi64(a, _mm512_set1_epi64(b3)));
        c4 = _mm512_sub_epi64(c4, _mm512_mullo_epi64(a, _mm512_set1_epi64(b4)));
        c5 = _mm512_sub_epi64(c5, _mm512_mullo_epi64(a, _mm512_set1_epi64(b5)));
        c6 = _mm512_sub_epi64(c6, _mm512_mullo_epi64(a, _mm512_set1_epi64(b6)));
        c7 = _mm512_sub_epi64(c7, _mm512_mullo_epi64(a, _mm512_set1_epi64(b7)));

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
}

void __micro_kernel_ntimes_ADBT(
    int64_t* __restrict A, int ldA, int64_t* __restrict B, int ldB,
    int64_t* __restrict C, int ldC, int k, int64_t* d)
{
    __m512i c0 = *(__m512i*)(C);
    __m512i c1 = *(__m512i*)(C + ldC);
    __m512i c2 = *(__m512i*)(C + 2 * ldC);
    __m512i c3 = *(__m512i*)(C + 3 * ldC);
    __m512i c4 = *(__m512i*)(C + 4 * ldC);
    __m512i c5 = *(__m512i*)(C + 5 * ldC);
    __m512i c6 = *(__m512i*)(C + 6 * ldC);
    __m512i c7 = *(__m512i*)(C + 7 * ldC);

    for (int p = 0; p < k; ++p) {
        __m512i a = *(__m512i*)(A + p * ldA);
        int64_t b0 = B[0];
        int64_t b1 = B[1];
        int64_t b2 = B[2];
        int64_t b3 = B[3];
        int64_t b4 = B[4];
        int64_t b5 = B[5];
        int64_t b6 = B[6];
        int64_t b7 = B[7];
        int64_t D = d[p];

        c0 = _mm512_sub_epi64(c0, _mm512_mullo_epi64(a, _mm512_set1_epi64(b0 * D)));
        c1 = _mm512_sub_epi64(c1, _mm512_mullo_epi64(a, _mm512_set1_epi64(b1 * D)));
        c2 = _mm512_sub_epi64(c2, _mm512_mullo_epi64(a, _mm512_set1_epi64(b2 * D)));
        c3 = _mm512_sub_epi64(c3, _mm512_mullo_epi64(a, _mm512_set1_epi64(b3 * D)));
        c4 = _mm512_sub_epi64(c4, _mm512_mullo_epi64(a, _mm512_set1_epi64(b4 * D)));
        c5 = _mm512_sub_epi64(c5, _mm512_mullo_epi64(a, _mm512_set1_epi64(b5 * D)));
        c6 = _mm512_sub_epi64(c6, _mm512_mullo_epi64(a, _mm512_set1_epi64(b6 * D)));
        c7 = _mm512_sub_epi64(c7, _mm512_mullo_epi64(a, _mm512_set1_epi64(b7 * D)));

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
}

#endif