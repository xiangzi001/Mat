
inline void __times(int* __restrict a, int b, int* __restrict ans, int len) {
    __m512i bb = _mm512_set1_epi32(b);

    int n = len / 16;
    int i;
    for (i = 0; i < n; ++i) {
        ((__m512i*)ans)[i] = _mm512_mullo_epi32(((__m512i*)a)[i], bb);
    }
    n = len - n * 16;
    if (n != 0) {
        ((__m512i*)ans)[i] = _mm512_mask_blend_epi32((1 << n) - 1, ((__m512i*)ans)[i],
                               _mm512_mullo_epi32(((__m512i*)a)[i], bb));
    }
}

inline void __add(int* __restrict a, int* __restrict b, int* __restrict ans, int len) {
    int n = len / 16;
    int i;
    for (i = 0; i < n; ++i) {
        ((__m512i*)ans)[i] = _mm512_add_epi32(((__m512i*)a)[i], ((__m512i*)b)[i]);
    }
    n = len - n * 16;
    if (n != 0) {
        ((__m512i*)ans)[i] = _mm512_mask_blend_epi32((1 << n) - 1, ((__m512i*)ans)[i],
                               _mm512_add_epi32(((__m512i*)a)[i], ((__m512i*)b)[i]));
    }
}

// ans = a + b * q
inline void __mixed(int* __restrict a, int* __restrict b, int q, int* __restrict ans, int len) {
    __m512i qq = _mm512_set1_epi32(q);
    int n = len / 16;
    int i;
    for (i = 0; i < n; ++i) {
        ((__m512i*)ans)[i] = _mm512_add_epi32(((__m512i*)a)[i],
                               _mm512_mullo_epi32(((__m512i*)b)[i], qq));
    }
    n = len - n * 16;
    if (n != 0) {
        ((__m512i*)ans)[i] = _mm512_mask_blend_epi32((1 << n) - 1, ((__m512i*)ans)[i],
                               _mm512_add_epi32(((__m512i*)a)[i],
                               _mm512_mullo_epi32(((__m512i*)b)[i], qq)));
    }
}

inline void __swap(int* __restrict a, int* __restrict b, int len) {
    __m512i temp;
    int n = len / 16;
    int i;
    for (i = 0; i < n; ++i) {
        temp = ((__m512i*)a)[i];
        ((__m512i*)a)[i] = ((__m512i*)b)[i];
        ((__m512i*)b)[i] = temp;
    }
    n = len - n * 16;
    if (n != 0) {
        __mmask16 mask = (1 << n) - 1;
        temp = ((__m512i*)a)[i];
        ((__m512i*)a)[i] = _mm512_mask_blend_epi32(mask, temp, ((__m512i*)b)[i]);
        ((__m512i*)b)[i] = _mm512_mask_blend_epi32(mask, ((__m512i*)b)[i], temp);
    }
}

inline int __dot(int* __restrict a, int* __restrict b, int len) {
    __m512i ans = _mm512_setzero_si512();
    int n = len / 16;
    int i;
    for (i = 0; i < n; ++i) {
        ans = _mm512_add_epi32(ans, _mm512_mullo_epi32(((__m512i*)a)[i], ((__m512i*)b)[i]));
    }
    n = len - n * 16;
    if (n != 0) {
        ans = _mm512_mask_blend_epi32((1 << n) - 1, ans,
               _mm512_add_epi32(ans, _mm512_mullo_epi32(((__m512i*)a)[i], ((__m512i*)b)[i])));
    }
    return _mm512_reduce_add_epi32(ans);
}

int __MaxAbs(int* a, int len) {
    int s = len / 16;
    __m512i max = _mm512_setzero_si512();
    for (int i = 0; i < s; ++i) {
        max = _mm512_max_epi32(max, _mm512_abs_epi32(((__m512i*)a)[i]));
    }
    len = len - s * 16;
    max = _mm512_max_epi32(max,
          _mm512_mask_blend_epi32((1 << len) - 1, _mm512_setzero_si512(),
          _mm512_abs_epi32(((__m512i*)a)[s])));
    return _mm512_reduce_max_epi32(max);
}

