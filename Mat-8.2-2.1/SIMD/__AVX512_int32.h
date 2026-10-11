inline void __ApplyGivensOnTheRight(int32_t* __restrict a, int32_t* __restrict b, int32_t c, int32_t s, int len) {
    __m512i ai, bi;
    __m512i cc(_mm512_set1_epi32(c)), ss(_mm512_set1_epi32(s));
    int n = len / 16;
    int i;
    for (i = 0; i < n; ++i) {
        ai = ((__m512i*)a)[i];
        bi = ((__m512i*)b)[i];
        ((__m512i*)a)[i] = _mm512_add_epi32(_mm512_mullo_epi32(ai, cc), _mm512_mullo_epi32(bi, ss));
        ((__m512i*)b)[i] = _mm512_sub_epi32(_mm512_mullo_epi32(bi, cc), _mm512_mullo_epi32(ai, ss));
    }

    n = len - n * 16;
    if (n != 0) {
        __mmask16 mask = (1 << n) - 1;
        ai = ((__m512i*)a)[i];
        bi = ((__m512i*)b)[i];
        ((__m512i*)a)[i] = _mm512_mask_blend_epi32(mask, ai, _mm512_add_epi32(_mm512_mullo_epi32(ai, cc), _mm512_mullo_epi32(bi, ss)));
        ((__m512i*)b)[i] = _mm512_mask_blend_epi32(mask, bi, _mm512_sub_epi32(_mm512_mullo_epi32(bi, cc), _mm512_mullo_epi32(ai, ss)));
    }
}

inline void __Schur_Householder_Right(
    int32_t* __restrict a, int32_t* __restrict b, int32_t* __restrict c,
    const int32_t* __restrict w, int len) {
    __m512i sum;
    __m512i w0(_mm512_set1_epi32(w[0])), w1(_mm512_set1_epi32(w[1])), w2(_mm512_set1_epi32(w[2]));

    int n = len / 16;
    int i;
    for (i = 0; i < n; ++i) {
        sum =
            _mm512_add_epi32(
                _mm512_mullo_epi32(((__m512i*)a)[i], w0),
                _mm512_add_epi32(
                    _mm512_mullo_epi32(((__m512i*)b)[i], w1),
                    _mm512_mullo_epi32(((__m512i*)c)[i], w2)
                )
            );
        ((__m512i*)a)[i] = _mm512_sub_epi32(((__m512i*)a)[i], _mm512_mullo_epi32(sum, w0));
        ((__m512i*)b)[i] = _mm512_sub_epi32(((__m512i*)b)[i], _mm512_mullo_epi32(sum, w1));
        ((__m512i*)c)[i] = _mm512_sub_epi32(((__m512i*)c)[i], _mm512_mullo_epi32(sum, w2));
    }
    n = len - n * 16;
    if (n != 0) {
        sum = _mm512_mask_blend_epi32((1 << n) - 1, _mm512_setzero_si512(),
            _mm512_add_epi32(
                _mm512_mullo_epi32(((__m512i*)a)[i], w0),
                _mm512_add_epi32(
                    _mm512_mullo_epi32(((__m512i*)b)[i], w1),
                    _mm512_mullo_epi32(((__m512i*)c)[i], w2)
                )
            ));
        ((__m512i*)a)[i] = _mm512_sub_epi32(((__m512i*)a)[i], _mm512_mullo_epi32(sum, w0));
        ((__m512i*)b)[i] = _mm512_sub_epi32(((__m512i*)b)[i], _mm512_mullo_epi32(sum, w1));
        ((__m512i*)c)[i] = _mm512_sub_epi32(((__m512i*)c)[i], _mm512_mullo_epi32(sum, w2));
    }
}

inline void __times(int32_t* __restrict a, int32_t b, int32_t* __restrict ans, int len) {
    __m512i bb(_mm512_set1_epi32(b));

    int n = len / 16;
    int i;
    for (i = 0; i < n; ++i) {
        ((__m512i*)ans)[i] = _mm512_mullo_epi32(((__m512i*)a)[i], bb);
    }
    n = len - n * 16;
    if (n != 0) {
        ((__m512i*)ans)[i] = _mm512_mask_blend_epi32((1 << n) - 1, ((__m512i*)ans)[i], _mm512_mullo_epi32(((__m512i*)a)[i], bb));
    }
}

inline void __add(int32_t* __restrict a, int32_t* __restrict b, int32_t* __restrict ans, int len) {
    int n = len / 16;
    int i;
    for (i = 0; i < n; ++i) {
        ((__m512i*)ans)[i] = _mm512_add_epi32(((__m512i*)a)[i], ((__m512i*)b)[i]);
    }
    n = len - n * 16;
    if (n != 0) {
        ((__m512i*)ans)[i] = _mm512_mask_blend_epi32((1 << n) - 1, ((__m512i*)ans)[i], _mm512_add_epi32(((__m512i*)a)[i], ((__m512i*)b)[i]));
    }
}

//ans=a+b*p
inline void __mixed(int32_t* __restrict a, int32_t* __restrict b, int32_t q, int32_t* __restrict ans, int len) {
    __m512i qq(_mm512_set1_epi32(q));
    int n = len / 16;
    int i;
    for (i = 0; i < n; ++i) {
        ((__m512i*)ans)[i] = _mm512_add_epi32(((__m512i*)a)[i], _mm512_mullo_epi32(((__m512i*)b)[i], qq));
    }
    n = len - n * 16;
    if (n != 0) {
        ((__m512i*)ans)[i] = _mm512_mask_blend_epi32((1 << n) - 1, ((__m512i*)ans)[i], _mm512_add_epi32(((__m512i*)a)[i], _mm512_mullo_epi32(((__m512i*)b)[i], qq)));
    }
}

inline void __swap(int32_t* __restrict a, int32_t* __restrict b, int len) {
    int n = len / 16;
    int i;
    for (i = 0; i < n; ++i) {
        __m512i temp = ((__m512i*)a)[i];
        ((__m512i*)a)[i] = ((__m512i*)b)[i];
        ((__m512i*)b)[i] = temp;
    }
    n = len - n * 16;
    if (n != 0) {
        __mmask16 mask = (1 << n) - 1;
        __m512i temp = ((__m512i*)a)[i];
        ((__m512i*)a)[i] = _mm512_mask_blend_epi32(mask, temp, ((__m512i*)b)[i]);
        ((__m512i*)b)[i] = _mm512_mask_blend_epi32(mask, ((__m512i*)b)[i], temp);
    }
}

inline int32_t __dot(int32_t* __restrict a, int32_t* __restrict b, int len) {
    __m512i ans0 = _mm512_setzero_si512();
    __m512i ans1 = _mm512_setzero_si512();
    __m512i ans2 = _mm512_setzero_si512();
    __m512i ans3 = _mm512_setzero_si512();
    int n = len / 16;
    int i = 0;
    for (; i <= n - 4; i += 4) {
        ans0 = _mm512_add_epi32(_mm512_mullo_epi32(((__m512i*)a)[i], ((__m512i*)b)[i]), ans0);
        ans1 = _mm512_add_epi32(_mm512_mullo_epi32(((__m512i*)a)[i + 1], ((__m512i*)b)[i + 1]), ans1);
        ans2 = _mm512_add_epi32(_mm512_mullo_epi32(((__m512i*)a)[i + 2], ((__m512i*)b)[i + 2]), ans2);
        ans3 = _mm512_add_epi32(_mm512_mullo_epi32(((__m512i*)a)[i + 3], ((__m512i*)b)[i + 3]), ans3);
    }
    for (; i < n; ++i) {
        ans0 = _mm512_add_epi32(_mm512_mullo_epi32(((__m512i*)a)[i], ((__m512i*)b)[i]), ans0);
    }
    __m512i ans = _mm512_add_epi32(_mm512_add_epi32(ans0, ans1), _mm512_add_epi32(ans2, ans3));

    n = len - n * 16;
    if (n != 0) {
        ans = _mm512_mask_blend_epi32((1 << n) - 1, ans, _mm512_add_epi32(_mm512_mullo_epi32(((__m512i*)a)[i], ((__m512i*)b)[i]), ans));
    }
    return _mm512_reduce_add_epi32(ans);
}

int32_t __MaxAbs(int32_t* a, int len) {
    int n = len / 16;
    __m512i max0 = _mm512_setzero_si512();
    __m512i max1 = _mm512_setzero_si512();
    __m512i max2 = _mm512_setzero_si512();
    __m512i max3 = _mm512_setzero_si512();
    int i = 0;
    for (; i <= n - 4; i += 4) {
        max0 = _mm512_max_epi32(max0, _mm512_abs_epi32(((__m512i*)a)[i]));
        max1 = _mm512_max_epi32(max1, _mm512_abs_epi32(((__m512i*)a)[i + 1]));
        max2 = _mm512_max_epi32(max2, _mm512_abs_epi32(((__m512i*)a)[i + 2]));
        max3 = _mm512_max_epi32(max3, _mm512_abs_epi32(((__m512i*)a)[i + 3]));
    }
    for (; i < n; ++i) {
        max0 = _mm512_max_epi32(max0, _mm512_abs_epi32(((__m512i*)a)[i]));
    }
    __m512i max = _mm512_max_epi32(_mm512_max_epi32(max0, max1), _mm512_max_epi32(max2, max3));

    n = len - n * 16;
    if (n != 0) {
        max = _mm512_max_epi32(max,
            _mm512_mask_blend_epi32((1 << n) - 1, _mm512_setzero_si512(), _mm512_abs_epi32(((__m512i*)a)[i]))
        );
    }
    return _mm512_reduce_max_epi32(max);
}