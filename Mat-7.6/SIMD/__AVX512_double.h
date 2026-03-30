
inline void __ApplyGivensOnTheRight(double* __restrict a, double* __restrict b, double c, double s, int len) {
    __m512d ai, bi;
	__m512d cc(_mm512_set1_pd(c)), ss(_mm512_set1_pd(s));
	int n = len / 8;
	int i;
    for (i = 0; i < n; ++i) {
        ai = ((__m512d*)a)[i];
        bi = ((__m512d*)b)[i];
		((__m512d*)a)[i] = _mm512_add_pd(_mm512_mul_pd(ai, cc), _mm512_mul_pd(bi, ss));
		((__m512d*)b)[i] = _mm512_sub_pd(_mm512_mul_pd(bi, cc), _mm512_mul_pd(ai, ss));
    }

	n = len - n * 8;
	if (n != 0) {
		__mmask8 mask = (1 << n) - 1;
		ai = ((__m512d*)a)[i];
		bi = ((__m512d*)b)[i];
		((__m512d*)a)[i] = _mm512_mask_blend_pd(mask, ai, _mm512_add_pd(_mm512_mul_pd(ai, cc), _mm512_mul_pd(bi, ss)));
		((__m512d*)b)[i] = _mm512_mask_blend_pd(mask, bi, _mm512_sub_pd(_mm512_mul_pd(bi, cc), _mm512_mul_pd(ai, ss)));
	}
}

inline void __Schur_HouseHolder_Right(
	double* __restrict a, double* __restrict b, double* __restrict c, 
	const double* __restrict w, int len) {
	__m512d sum; 
	__m512d w0(_mm512_set1_pd(w[0])), w1(_mm512_set1_pd(w[1])), w2(_mm512_set1_pd(w[2]));

	int n = len / 8;
	int i;
	for (i = 0; i < n; ++i) {
		sum = _mm512_add_pd(_mm512_add_pd(
			_mm512_mul_pd(((__m512d*)a)[i], w0), 
			_mm512_mul_pd(((__m512d*)b)[i], w1) ), 
			_mm512_mul_pd(((__m512d*)c)[i], w2) );
		((__m512d*)a)[i] = _mm512_sub_pd(((__m512d*)a)[i], _mm512_mul_pd(sum, w0));
		((__m512d*)b)[i] = _mm512_sub_pd(((__m512d*)b)[i], _mm512_mul_pd(sum, w1));
		((__m512d*)c)[i] = _mm512_sub_pd(((__m512d*)c)[i], _mm512_mul_pd(sum, w2));
	}
	n = len - n * 8;
	if (n != 0) {
		sum = _mm512_mask_blend_pd((1 << n) - 1, _mm512_setzero_pd(), _mm512_add_pd(_mm512_add_pd(
			_mm512_mul_pd(((__m512d*)a)[i], w0),
			_mm512_mul_pd(((__m512d*)b)[i], w1)),
			_mm512_mul_pd(((__m512d*)c)[i], w2)));
		((__m512d*)a)[i] = _mm512_sub_pd(((__m512d*)a)[i], _mm512_mul_pd(sum, w0));
		((__m512d*)b)[i] = _mm512_sub_pd(((__m512d*)b)[i], _mm512_mul_pd(sum, w1));
		((__m512d*)c)[i] = _mm512_sub_pd(((__m512d*)c)[i], _mm512_mul_pd(sum, w2));
	}
}

inline void __times(double* __restrict a, double b, double* __restrict ans, int len) {
	__m512d bb(_mm512_set1_pd(b));

	int n = len / 8;
	int i;
	for (i = 0; i < n; ++i) {
		((__m512d*)ans)[i] = _mm512_mul_pd(((__m512d*)a)[i], bb);
	}
	n = len - n * 8;
	if (n != 0) {
		((__m512d*)ans)[i] = _mm512_mask_blend_pd((1 << n) - 1, ((__m512d*)ans)[i], _mm512_mul_pd(((__m512d*)a)[i], bb));
	}
}

inline void __add(double* __restrict a, double* __restrict b, double* __restrict ans, int len) {
	int n = len / 8;
	int i;
	for (i = 0; i < n; ++i) {
		((__m512d*)ans)[i] = _mm512_add_pd(((__m512d*)a)[i], ((__m512d*)b)[i]);
	}
	n = len - n * 8;
	if (n != 0) {
		((__m512d*)ans)[i] = _mm512_mask_blend_pd((1 << n) - 1, ((__m512d*)ans)[i], _mm512_add_pd(((__m512d*)a)[i], ((__m512d*)b)[i]));
	}
}

//ans=a+b*p
inline void __mixed(double* __restrict a, double* __restrict b, double q, double* __restrict ans, int len) {
	__m512d qq(_mm512_set1_pd(q));
	int n = len / 8;
	int i;
	for (i = 0; i < n; ++i) {
		((__m512d*)ans)[i] = _mm512_add_pd(((__m512d*)a)[i], _mm512_mul_pd(((__m512d*)b)[i], qq));
	}
	n = len - n * 8;
	if (n != 0) {
		((__m512d*)ans)[i] = _mm512_mask_blend_pd((1 << n) - 1, ((__m512d*)ans)[i], _mm512_add_pd(((__m512d*)a)[i], _mm512_mul_pd(((__m512d*)b)[i], qq)));
	}
}

inline void __swap(double* __restrict a, double* __restrict b, int len) {
	__m512d temp;
	int n = len / 8;
	int i;
	for (i = 0; i < n; ++i) {
		temp = ((__m512d*)a)[i];
		((__m512d*)a)[i] = ((__m512d*)b)[i];
		((__m512d*)b)[i] = temp;
	}
	n = len - n * 8;
	if (n != 0) {
		__mmask8 mask = (1 << n) - 1;
		temp = ((__m512d*)a)[i];
		((__m512d*)a)[i] = _mm512_mask_blend_pd(mask, temp, ((__m512d*)b)[i]);
		((__m512d*)b)[i] = _mm512_mask_blend_pd(mask, ((__m512d*)b)[i], temp);
	}
}

inline double __dot(double* __restrict a, double* __restrict b, int len) {
	__m512d ans(_mm512_setzero_pd());
	int n = len / 8;
	int i;
	for (i = 0; i < n; ++i) {
		ans = _mm512_add_pd(ans, _mm512_mul_pd(((__m512d*)a)[i], ((__m512d*)b)[i]));
	}
	n = len - n * 8;
	if (n != 0) {
		ans = _mm512_mask_blend_pd((1 << n) - 1, ans, _mm512_add_pd(ans, _mm512_mul_pd(((__m512d*)a)[i], ((__m512d*)b)[i])));
	}

	return _mm512_reduce_add_pd(ans);
}
