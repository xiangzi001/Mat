
template <typename T>
double Norm_one(const MAT<T>& A) {
	double ret(0);
	double temp;
	int i;
	T* a;
	for (int j = 0; j < A.cols; ++j) {
		temp = 0;
		a = A.data + A.rows * j;
		for (i = 0; i < A.rows; ++i) {
			temp += fabs(a[i]);
		}
		if (temp > ret)
			ret = temp;
	}

	return ret;
}

template <typename T>
double Norm_inf(const MAT<T>& A) {
	double ret(0);
	double temp;
	int j;
	for (int i = 0; i < A.rows; ++i) {
		temp = 0;
		for (j = 0; j < A.cols; ++j) {
			temp += fabs(A.data[j * A.rows + i]);
		}
		if (temp > ret)
			ret = temp;
	}

	return ret;
}

template <typename T>
double Norm_F(const MAT<T>& A) {
	return sqrt(__dot(A.data, A.data, A.rows * A.cols));
}

template <typename T>
auto MaxAbs(const MAT<T>& A) {
	return __MaxAbs(A.data, A.rows * A.cols);
}

#ifdef XiangZiMatImport_SIMD

double Norm_one(const MAT<double>& A) {
	int i, j;
	int n = A.rows / 8;
	int m = A.rows - n * 8;

	double max = 0;
	double t;

	__m512d sum;
	__m512d* temp;
	__mmask8 mask = (1 << m) - 1;

	for (i = 0; i < A.cols; ++i) {
		sum = _mm512_mask_blend_pd(mask, _mm512_setzero_pd(), _mm512_abs_pd(*(__m512d*)(A.data + i * A.rows)));
		temp = (__m512d*)(A.data + i * A.rows + m);
		for (j = 0; j < n; ++j) {
			sum = _mm512_add_pd(sum, _mm512_abs_pd(temp[j]));
		}
		t = _mm512_reduce_add_pd(sum);
		if (t > max)
			max = t;
	}
	return max;
}

double Norm_inf(const MAT<double>& A) {
	int i;
	int n = (int)(A.rows / 8) * 8;
	__m512d max = _mm512_setzero_pd();
	__m512d sum;

	for (int j = 0; j < n; j += 8) {
		sum = _mm512_setzero_pd();
		for (i = 0; i < A.cols; ++i) {
			sum = _mm512_add_pd(sum, _mm512_abs_pd(*(__m512d*)(A.data + i * A.rows + j)));
		}
		max = _mm512_max_pd(max, sum);
	}

	sum = _mm512_setzero_pd();
	for (i = 0; i < A.cols; ++i) {
		sum = _mm512_add_pd(sum, _mm512_abs_pd(*(__m512d*)(A.data + i * A.rows + n)));
	}
	max = _mm512_max_pd(max, _mm512_mask_blend_pd((1 << (A.rows - n)) - 1, _mm512_setzero_pd(), sum));

	return _mm512_reduce_max_pd(max);
}

#endif
