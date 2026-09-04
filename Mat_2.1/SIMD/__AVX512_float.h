inline void __ApplyGivensOnTheRight(float* __restrict a, float* __restrict b, float c, float s, int len) {
    __m512 ai, bi;
    __m512 cc(_mm512_set1_ps(c)), ss(_mm512_set1_ps(s));
    int n = len / 16;
    int i;
    for (i = 0; i < n; ++i) {
        ai = ((__m512*)a)[i];
        bi = ((__m512*)b)[i];
        ((__m512*)a)[i] = _mm512_fmadd_ps(ai, cc, _mm512_mul_ps(bi, ss));
        ((__m512*)b)[i] = _mm512_fnmadd_ps(ai, ss, _mm512_mul_ps(bi, cc));
    }

    n = len - n * 16;
    if (n != 0) {
        __mmask16 mask = (1 << n) - 1;
        ai = ((__m512*)a)[i];
        bi = ((__m512*)b)[i];
        ((__m512*)a)[i] = _mm512_mask_blend_ps(mask, ai, _mm512_fmadd_ps(ai, cc, _mm512_mul_ps(bi, ss)));
        ((__m512*)b)[i] = _mm512_mask_blend_ps(mask, bi, _mm512_fnmadd_ps(ai, ss, _mm512_mul_ps(bi, cc)));
    }
}

inline void __Schur_Householder_Right(
    float* __restrict a, float* __restrict b, float* __restrict c,
    const float* __restrict w, int len) {
    __m512 sum;
    __m512 w0(_mm512_set1_ps(w[0])), w1(_mm512_set1_ps(w[1])), w2(_mm512_set1_ps(w[2]));

    int n = len / 16;
    int i;
    for (i = 0; i < n; ++i) {
        sum =
            _mm512_fmadd_ps(((__m512*)a)[i], w0,
                _mm512_fmadd_ps(((__m512*)b)[i], w1,
                    _mm512_mul_ps(((__m512*)c)[i], w2)));
        ((__m512*)a)[i] = _mm512_fnmadd_ps(sum, w0, ((__m512*)a)[i]);
        ((__m512*)b)[i] = _mm512_fnmadd_ps(sum, w1, ((__m512*)b)[i]);
        ((__m512*)c)[i] = _mm512_fnmadd_ps(sum, w2, ((__m512*)c)[i]);
    }
    n = len - n * 16;
    if (n != 0) {
        sum = _mm512_mask_blend_ps((1 << n) - 1, _mm512_setzero_ps(),
            _mm512_fmadd_ps(((__m512*)a)[i], w0,
                _mm512_fmadd_ps(((__m512*)b)[i], w1,
                    _mm512_mul_ps(((__m512*)c)[i], w2))));
        ((__m512*)a)[i] = _mm512_fnmadd_ps(sum, w0, ((__m512*)a)[i]);
        ((__m512*)b)[i] = _mm512_fnmadd_ps(sum, w1, ((__m512*)b)[i]);
        ((__m512*)c)[i] = _mm512_fnmadd_ps(sum, w2, ((__m512*)c)[i]);
    }
}

inline void __times(float* __restrict a, float b, float* __restrict ans, int len) {
    __m512 bb(_mm512_set1_ps(b));

    int n = len / 16;
    int i;
    for (i = 0; i < n; ++i) {
        ((__m512*)ans)[i] = _mm512_mul_ps(((__m512*)a)[i], bb);
    }
    n = len - n * 16;
    if (n != 0) {
        ((__m512*)ans)[i] = _mm512_mask_blend_ps((1 << n) - 1, ((__m512*)ans)[i], _mm512_mul_ps(((__m512*)a)[i], bb));
    }
}

inline void __add(float* __restrict a, float* __restrict b, float* __restrict ans, int len) {
    int n = len / 16;
    int i;
    for (i = 0; i < n; ++i) {
        ((__m512*)ans)[i] = _mm512_add_ps(((__m512*)a)[i], ((__m512*)b)[i]);
    }
    n = len - n * 16;
    if (n != 0) {
        ((__m512*)ans)[i] = _mm512_mask_blend_ps((1 << n) - 1, ((__m512*)ans)[i], _mm512_add_ps(((__m512*)a)[i], ((__m512*)b)[i]));
    }
}

//ans=a+b*p
inline void __mixed(float* __restrict a, float* __restrict b, float q, float* __restrict ans, int len) {
    __m512 qq(_mm512_set1_ps(q));
    int n = len / 16;
    int i;
    for (i = 0; i < n; ++i) {
        ((__m512*)ans)[i] = _mm512_fmadd_ps(((__m512*)b)[i], qq, ((__m512*)a)[i]);
    }
    n = len - n * 16;
    if (n != 0) {
        ((__m512*)ans)[i] = _mm512_mask_blend_ps((1 << n) - 1, ((__m512*)ans)[i], _mm512_fmadd_ps(((__m512*)b)[i], qq, ((__m512*)a)[i]));
    }
}

inline void __swap(float* __restrict a, float* __restrict b, int len) {
    int n = len / 16;
    int i;
    for (i = 0; i < n; ++i) {
        __m512 temp = ((__m512*)a)[i];
        ((__m512*)a)[i] = ((__m512*)b)[i];
        ((__m512*)b)[i] = temp;
    }
    n = len - n * 16;
    if (n != 0) {
        __mmask16 mask = (1 << n) - 1;
        __m512 temp = ((__m512*)a)[i];
        ((__m512*)a)[i] = _mm512_mask_blend_ps(mask, temp, ((__m512*)b)[i]);
        ((__m512*)b)[i] = _mm512_mask_blend_ps(mask, ((__m512*)b)[i], temp);
    }
}

inline float __dot(float* __restrict a, float* __restrict b, int len) {
    __m512 ans0 = _mm512_setzero_ps();
    __m512 ans1 = _mm512_setzero_ps();
    __m512 ans2 = _mm512_setzero_ps();
    __m512 ans3 = _mm512_setzero_ps();
    int n = len / 16;
    int i = 0;
    for (; i <= n - 4; i += 4) {
        ans0 = _mm512_fmadd_ps(((__m512*)a)[i], ((__m512*)b)[i], ans0);
        ans1 = _mm512_fmadd_ps(((__m512*)a)[i + 1], ((__m512*)b)[i + 1], ans1);
        ans2 = _mm512_fmadd_ps(((__m512*)a)[i + 2], ((__m512*)b)[i + 2], ans2);
        ans3 = _mm512_fmadd_ps(((__m512*)a)[i + 3], ((__m512*)b)[i + 3], ans3);
    }
    for (; i < n; ++i) {
        ans0 = _mm512_fmadd_ps(((__m512*)a)[i], ((__m512*)b)[i], ans0);
    }
    __m512 ans = _mm512_add_ps(_mm512_add_ps(ans0, ans1), _mm512_add_ps(ans2, ans3));

    n = len - n * 16;
    if (n != 0) {
        ans = _mm512_mask_blend_ps((1 << n) - 1, ans, _mm512_fmadd_ps(((__m512*)a)[i], ((__m512*)b)[i], ans));
    }
    return _mm512_reduce_add_ps(ans);
}

float __MaxAbs(float* a, int len) {
    int n = len / 16;
    __m512 max0 = _mm512_setzero_ps();
    __m512 max1 = _mm512_setzero_ps();
    __m512 max2 = _mm512_setzero_ps();
    __m512 max3 = _mm512_setzero_ps();
    int i = 0;
    for (; i <= n - 4; i += 4) {
        max0 = _mm512_max_ps(max0, _mm512_abs_ps(((__m512*)a)[i]));
        max1 = _mm512_max_ps(max1, _mm512_abs_ps(((__m512*)a)[i + 1]));
        max2 = _mm512_max_ps(max2, _mm512_abs_ps(((__m512*)a)[i + 2]));
        max3 = _mm512_max_ps(max3, _mm512_abs_ps(((__m512*)a)[i + 3]));
    }
    for (; i < n; ++i) {
        max0 = _mm512_max_ps(max0, _mm512_abs_ps(((__m512*)a)[i]));
    }
    __m512 max = _mm512_max_ps(_mm512_max_ps(max0, max1), _mm512_max_ps(max2, max3));

    n = len - n * 16;
    if (n != 0) {
        max = _mm512_max_ps(max,
            _mm512_mask_blend_ps((1 << n) - 1, _mm512_setzero_ps(), _mm512_abs_ps(((__m512*)a)[i]))
        );
    }
    return _mm512_reduce_max_ps(max);
}