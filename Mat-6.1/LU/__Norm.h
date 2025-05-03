
template <typename T>
double Norm_one(const MAT<T>& A) {
	double ret(0);
	double temp;
	int i;
	for (int j = 0; j < A.cols; ++j) {
		temp = 0;
		for (i = 0; i < A.rows; ++i) {
			temp += fabs(A.data[i * A.cols + j]);
		}
		if (temp > ret)
			ret = temp;
	}

	return ret;
}

double Norm_one(const MAT<double>& A) {
	int i, j;
	int n = (int)(A.cols / 8) * 8;
	__m512d max = _mm512_setzero_pd();
	__m512d sum;

	for (j = 0; j < n; j += 8) {
		sum = _mm512_setzero_pd();
		for (i = 0; i < A.rows; ++i) {
			sum = _mm512_add_pd(sum, _mm512_abs_pd(*(__m512d*)(A.data + i * A.cols + j)));
		}
		max = _mm512_max_pd(max, sum);
	}

	sum = _mm512_setzero_pd();
	for (i = 0; i < A.rows; ++i) {
		sum = _mm512_add_pd(sum, _mm512_abs_pd(*(__m512d*)(A.data + i * A.cols + j)));
	}
	max = _mm512_max_pd(max, _mm512_mask_blend_pd((1 << (A.cols - n)) - 1, _mm512_setzero_pd(), sum));

	return _mm512_reduce_max_pd(max);
}
float Norm_one(const MAT<float>& A) {
	int i, j;
	int n = (int)(A.cols / 16) * 16;
	__m512 max = _mm512_setzero_ps();
	__m512 sum;

	for (j = 0; j < n; j += 16) {
		sum = _mm512_setzero_ps();
		for (i = 0; i < A.rows; ++i) {
			sum = _mm512_add_ps(sum, _mm512_abs_ps(*(__m512*)(A.data + i * A.cols + j)));
		}
		max = _mm512_max_ps(max, sum);
	}

	sum = _mm512_setzero_ps();
	for (i = 0; i < A.rows; ++i) {
		sum = _mm512_add_ps(sum, _mm512_abs_ps(*(__m512*)(A.data + i * A.cols + j)));
	}
	max = _mm512_max_ps(max, _mm512_mask_blend_ps((1 << (A.cols - n)) - 1, _mm512_setzero_ps(), sum));

	return _mm512_reduce_max_ps(max);
}

template <typename T>
double Norm_inf(const MAT<T>& A) {
	double ret(0);
	double temp;
	int j;
	for (int i = 0; i < A.rows; ++i) {
		temp = 0;
		for (j = 0; j < A.cols; ++j) {
			temp += fabs(A.data[i * A.cols + j]);
		}
		if (temp > ret)
			ret = temp;
	}

	return ret;
}

double Norm_inf(const MAT<double>& A) {
	int i, j;
	int n = A.cols / 8;
	int m = A.cols - n * 8;

	double max = 0;
	double t;

	__m512d sum;
	__m512d* temp;
	__mmask8 mask = (1 << m) - 1;

	for (i = 0; i < A.rows; ++i) {
		sum = _mm512_mask_blend_pd(mask, _mm512_setzero_pd(), _mm512_abs_pd(*(__m512d*)(A.data + i * A.cols)));
		temp = (__m512d*)(A.data + i * A.cols + m);
		for (j = 0; j < n; ++j) {
			sum = _mm512_add_pd(sum, _mm512_abs_pd(temp[j]));
		}
		t = _mm512_reduce_add_pd(sum);
		if (t > max)
			max = t;
	}
	return max;
}

float Norm_inf(const MAT<float>& A) {
	int i, j;
	int n = A.cols / 16;
	int m = A.cols - n * 16;

	float max = 0;
	float t;

	__m512 sum;
	__m512* temp;
	__mmask16 mask = (1 << m) - 1;

	for (i = 0; i < A.rows; ++i) {
		sum = _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), _mm512_abs_ps(*(__m512*)(A.data + i * A.cols)));
		temp = (__m512*)(A.data + i * A.cols + m);
		for (j = 0; j < n; ++j) {
			sum = _mm512_add_ps(sum, _mm512_abs_ps(temp[j]));
		}
		t = _mm512_reduce_add_ps(sum);
		if (t > max)
			max = t;
	}
	return max;
}

template <typename T>
double Norm_F(const MAT<T>& A) {
	double ret = 0;
	for (int i = 0; i < A.rows*A.cols; ++i) {
		ret += A.data[i] * A.data[i];
	}

	return sqrt(ret);
}

double Norm_F(const MAT<double>& A) {
	long long s = A.rows * A.cols;
	long long n = s / 8;
	int m = s - n * 8;

	__m512d sum = _mm512_mask_blend_pd((1 << m) - 1, _mm512_setzero_pd(), _mm512_mul_pd(*(__m512d*)(A.data), *(__m512d*)(A.data)));
	__m512d* temp = (__m512d*)(A.data + m);
	for (long long i = 0; i < n; ++i) {
		sum = _mm512_add_pd(sum, _mm512_mul_pd(temp[i], temp[i]));
	}

	return sqrt(_mm512_reduce_add_pd(sum));
}
double Norm_F(const MAT<float>& A) {
	long long s = (long long)A.rows * A.cols;
	long long n = s / 16;
	int m = s - n * 16;

	__m512 sum = _mm512_mask_blend_ps((1 << m) - 1, _mm512_setzero_ps(), _mm512_mul_ps(*(__m512*)(A.data), *(__m512*)(A.data)));
	__m512* temp = (__m512*)(A.data + m);
	for (long long i = 0; i < n; ++i) {
		sum = _mm512_add_ps(sum, _mm512_mul_ps(temp[i], temp[i]));
	}

	return sqrt(_mm512_reduce_add_ps(sum));
}