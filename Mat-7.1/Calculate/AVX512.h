
inline void __times(double* a, double b, double* ans, int m) {
	int n = m / 8;
	m = m - n * 8;
	__m512d mul = _mm512_set1_pd(b);

	if (m) {
		*(__m512d*)ans = _mm512_mask_blend_pd((1 << m) - 1, *(__m512d*)ans, _mm512_mul_pd(mul, *(__m512d*)a));
		a = a + m;
		ans = ans + m;
	}

	for (m = 0; m < n; ++m) {
		((__m512d*)(ans))[m] = _mm512_mul_pd(mul, ((__m512d*)(a))[m]);
	}
}

inline void __add(double* a, double* b, double* ans, int m) {
	int n = m / 8;
	m = m - n * 8;

	if (m) {
		*(__m512d*)ans = _mm512_mask_blend_pd((1 << m) - 1, *(__m512d*)ans, _mm512_add_pd(*(__m512d*)a, *(__m512d*)b));
		a = a + m;
		b = b + m;
		ans = ans + m;
	}

	for (m = 0; m < n; ++m) {
		((__m512d*)(ans))[m] = _mm512_add_pd(((__m512d*)(a))[m], ((__m512d*)(b))[m]);
	}
}

inline void __sub(double* a, double* b, double* ans, int m) {
	int n = m / 8;
	m = m - n * 8;

	if (m)
		*(__m512d*)ans = _mm512_mask_blend_pd((1 << m) - 1, *(__m512d*)ans, _mm512_sub_pd(*(__m512d*)a, *(__m512d*)b));

	for (m = 0; m < n; ++m) {
		((__m512d*)(ans))[m] = _mm512_sub_pd(((__m512d*)(a))[m], ((__m512d*)(b))[m]);
	}
}

inline void __mixed1(double* a, double* b, double t, int m) {
	int n = m / 8;
	m = m - n * 8;
	__m512d temp1;
	__m512d temp2 = _mm512_set1_pd(t);

	if (m) {
		temp1 = _mm512_mask_blend_pd((1 << m) - 1, *(__m512d*)b, _mm512_mul_pd(*(__m512d*)a, temp2));
		*(__m512d*)a = _mm512_mask_blend_pd((1 << m) - 1, *(__m512d*)a, *(__m512d*)b);
		*(__m512d*)b = temp1;
		a = a + m;
		b = b + m;
	}

	for (m = 0; m < n; ++m) {
		temp1 = _mm512_mul_pd(((__m512d*)a)[m], temp2);
		((__m512d*)a)[m] = ((__m512d*)b)[m];
		((__m512d*)b)[m] = temp1;
	}
}

inline void __mixed2(double* a, double p, double* b, double q, double* ans, int m) {
	int n = m / 8;
	m = m - n * 8;
	__m512d P = _mm512_set1_pd(p);
	__m512d Q = _mm512_set1_pd(q);

	if (m) {
		*(__m512d*)ans = _mm512_mask_blend_pd((1 << m) - 1, *(__m512d*)ans, _mm512_add_pd(
			_mm512_mul_pd(*(__m512d*)a, P), _mm512_mul_pd(*(__m512d*)b, Q)
		));
	}

	for (m = 0; m < n; ++m) {
		((__m512d*)(ans))[m] = _mm512_add_pd(_mm512_mul_pd(((__m512d*)(a))[m], P), _mm512_mul_pd(((__m512d*)(b))[m], Q));
	}
}

inline void __mixed3(double* a, double* b, double q, double* ans, int m) {
	int n = m / 8;
	m = m - n * 8;
	__m512d Q = _mm512_set1_pd(q);

	if (m) {
		*(__m512d*)ans = _mm512_mask_blend_pd((1 << m) - 1, *(__m512d*)ans, _mm512_add_pd(
			*(__m512d*)a, _mm512_mul_pd(*(__m512d*)b, Q)
		));
		a = a + m;
		b = b + m;
		ans = ans + m;
	}

	for (m = 0; m < n; ++m) {
		((__m512d*)ans)[m] = _mm512_add_pd(((__m512d*)a)[m], _mm512_mul_pd(((__m512d*)b)[m], Q));
	}
}

inline void __swap(double* a, double* b, int m) {
	int n = m / 8;
	m = m - n * 8;
	__m512d temp1;

	if (m) {
		temp1 = _mm512_mask_blend_pd((1 << m) - 1, *(__m512d*)b, *(__m512d*)a);
		*(__m512d*)a = _mm512_mask_blend_pd((1 << m) - 1, *(__m512d*)a, *(__m512d*)b);
		*(__m512d*)b = temp1;
		a = a + m;
		b = b + m;
	}

	for (m = 0; m < n; ++m) {
		temp1 = ((__m512d*)a)[m];
		((__m512d*)a)[m] = ((__m512d*)b)[m];
		((__m512d*)b)[m] = temp1;
	}
}

inline double __dot(double* a, double* b, int m) {
	__m512d ans;
	int n = m / 8;
	m = m - n * 8;

	if (m) {
		ans = _mm512_mask_blend_pd((1 << m) - 1, _mm512_setzero_pd(), _mm512_mul_pd(*(__m512d*)a, *(__m512d*)b));
		a = a + m;
		b = b + m;
	}
	else {
		ans = _mm512_setzero_pd();
	}

	for (m = 0; m < n; ++m) {
		ans = _mm512_add_pd(ans, _mm512_mul_pd(((__m512d*)a)[m], ((__m512d*)b)[m]));
	}

	return _mm512_reduce_add_pd(ans);
}

inline double __dot3(double* a, double* b, double* c, int l) {
	__m512d ans;
	int n = l / 8;
	int m;

	ans = _mm512_setzero_pd();
	for (m = 0; m < n; ++m) {
		ans = _mm512_add_pd(ans, _mm512_mul_pd(_mm512_mul_pd(((__m512d*)a)[m], ((__m512d*)b)[m]), ((__m512d*)c)[m]));
	}

	m = l - n * 8;
	if (m) {
		ans =
			_mm512_add_pd(ans, _mm512_mask_blend_pd((1 << m) - 1, _mm512_setzero_pd(),
				_mm512_mul_pd(_mm512_mul_pd(((__m512d*)a)[n], ((__m512d*)b)[n]), ((__m512d*)c)[n])
			));
	}
	return _mm512_reduce_add_pd(ans);
}
