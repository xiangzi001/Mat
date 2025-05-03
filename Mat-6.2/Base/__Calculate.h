
template<typename T>
void times_A_B(MAT<T>& A, MAT<T>& B, MAT<T>& C, T* w) {
	int i, j, k;
	T v;
	for (j = 0; j < C.cols; ++j) {
		for (k = 0; k < B.rows; ++k) {
			w[k] = B[k][j];
		}
		for (i = 0; i < A.rows; ++i) {
			v = 0;
			for (k = 0; k < A.cols; ++k) {
				v += A[i][k] * w[k];
			}
			C[i][j] = v;
		}
	}
}

void times_A_B(MAT<double>& A, MAT<double>& B, MAT<double>& C, double* w) {
	int i, j, k;
	int n = A.cols / 8;
	int m = A.cols - n * 8;

	__m512d add;
	__m512d* tempA;
	__m512d* tempB = (__m512d*)(w + m);

	__mmask8 mask = (1 << m) - 1;

	for (j = 0; j < C.cols; ++j) {
		for (k = 0; k < B.rows; ++k) {
			w[k] = B[k][j];
		}
		for (i = 0; i < C.rows; ++i) {
			add = _mm512_mask_blend_pd(mask, _mm512_setzero_pd(), _mm512_mul_pd( *(__m512d*)(A[i]), *(__m512d*)(w) ));
			tempA = (__m512d*)(A[i] + m);
			for (k = 0; k < n; ++k) {
				add = _mm512_add_pd(add, _mm512_mul_pd(tempA[k], tempB[k]));
			}
			C[i][j] = _mm512_reduce_add_pd(add);
		}
	}
}
void times_A_B(MAT<float>& A, MAT<float>& B, MAT<float>& C, float* w) {
	int i, j, k;
	int n = A.cols / 16;
	int m = A.cols - n * 16;

	__m512 add;
	__m512* tempA;
	__m512* tempB = (__m512*)(w + m);

	__mmask16 mask = (1 << m) - 1;

	for (j = 0; j < C.cols; ++j) {
		for (k = 0; k < B.rows; ++k) {
			w[k] = B[k][j];
		}
		for (i = 0; i < C.rows; ++i) {
			add = _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), _mm512_mul_ps(*(__m512*)(A[i]), *(__m512*)(w)));
			tempA = (__m512*)(A[i] + m);
			for (k = 0; k < n; ++k) {
				add = _mm512_add_ps(add, _mm512_mul_ps(tempA[k], tempB[k]));
			}
			C[i][j] = _mm512_reduce_add_ps(add);
		}
	}
}

template<typename T>
void times_A_B(const MAT<T>& A, const MAT<T>& B, const MAT<T>& C) {
	int i, j, k = C.rows * C.cols;
	for (i = 0; i < k; ++i) {
		C.data[i] = 0;
	}
	T v;
	for (i = 0; i < C.rows; ++i) {
		for (k = 0; k < A.cols; ++k) {
			v = (A.data + A.cols * i)[k];
			for (j = 0; j < C.cols; ++j)
				(C.data + C.cols * i)[j] += v * (B.data + B.cols * k)[j];
		}
	}
}
void times_A_B(const MAT<double>& A, const MAT<double>& B, const MAT<double>& C) {
	int i, j, k;
	ZeroMemory(C.data, sizeof(double) * C.rows * C.cols);
	int n = C.cols / 8;
	int m = C.cols - n * 8;
	__m512d mul;
	__m512d* tempB;
	__m512d* tempC;

	__mmask8 mask = (1 << m) - 1;

	for (i = 0; i < C.rows; ++i) {
		tempC = (__m512d*)(C.data + C.cols * i + m);
		for (k = 0; k < A.cols; ++k) {
			mul = _mm512_set1_pd((A.data + A.cols * i)[k]);
			*(__m512d*)(C.data + C.cols * i) = _mm512_add_pd(*(__m512d*)(C.data + C.cols * i), _mm512_mask_blend_pd(mask, _mm512_setzero_pd(), _mm512_mul_pd(mul, *(__m512d*)(B.data + B.cols * k))));
			tempB = (__m512d*)(B.data + B.cols * k + m);
			for (j = 0; j < n; ++j)
				tempC[j] = _mm512_add_pd(tempC[j], _mm512_mul_pd(mul, tempB[j]));
		}
	}
}
void times_A_B(const MAT<float>& A, const MAT<float>& B, const MAT<float>& C) {
	int i, j, k;
	ZeroMemory(C.data, sizeof(float) * C.rows * C.cols);
	int n = C.cols / 16;
	int m = C.cols - n * 16;
	__m512 mul;
	__m512* tempB;
	__m512* tempC;

	__mmask16 mask = (1 << m) - 1;

	for (i = 0; i < C.rows; ++i) {
		tempC = (__m512*)(C.data + C.cols * i + m);
		for (k = 0; k < A.cols; ++k) {
			mul = _mm512_set1_ps((A.data + A.cols * i)[k]);
			*(__m512*)(C.data + C.cols * i) = _mm512_add_ps(*(__m512*)(C.data + C.cols * i), _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), _mm512_mul_ps(mul, *(__m512*)(B.data + B.cols * k))));
			tempB = (__m512*)(B.data + B.cols * k + m);
			for (j = 0; j < n; ++j)
				tempC[j] = _mm512_add_ps(tempC[j], _mm512_mul_ps(mul, tempB[j]));
		}
	}
}


template<typename T>
void times_A_BT(MAT<T>& A, MAT<T>& B, MAT<T>& C) {
	int i, j, k;
	for (i = 0; i < C.rows; ++i) {
		for (j = 0; j < C.rows; ++j) {
			C[i][j] = 0;
			for (k = 0; k < A.cols; ++k)
				C[i][j] += A[i][k] * B[j][k];
		}
	}
}

void times_A_BT(MAT<double>& A, MAT<double>& B, MAT<double>& C) {
	int i, j, k;
	int n = A.cols / 8;
	int m = A.cols - n * 8;
	__m512d add;
	__m512d* tempA;
	__m512d* tempB;

	__mmask8 mask = (1 << m) - 1;

	for (i = 0; i < C.rows; ++i) {
		for (j = 0; j < C.rows; ++j) {
			add = _mm512_mask_blend_pd(mask, _mm512_setzero_pd(), _mm512_mul_pd(*(__m512d*)(A[i]), *(__m512d*)(B[j])));
			tempA = (__m512d*)(A[i] + m);
			tempB = (__m512d*)(B[j] + m);
			for (k = 0; k < n;  ++k) {
				add = _mm512_add_pd(add, _mm512_mul_pd(tempA[k], tempB[k]));
			}
			C[i][j] = _mm512_reduce_add_pd(add);
		}
	}
}
void times_A_BT(MAT<float>& A, MAT<float>& B, MAT<float>& C) {
	int i, j, k;
	int n = A.cols / 16;
	int m = A.cols - n * 16;
	__m512 add;
	__m512* tempA;
	__m512* tempB;

	__mmask16 mask = (1 << m) - 1;

	for (i = 0; i < C.rows; ++i) {
		for (j = 0; j < C.rows; ++j) {
			add = _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), _mm512_mul_ps(*(__m512*)(A[i]), *(__m512*)(B[j])));
			tempA = (__m512*)(A[i] + m);
			tempB = (__m512*)(B[j] + m);
			for (k = 0; k < n; ++k) {
				add = _mm512_add_ps(add, _mm512_mul_ps(tempA[k], tempB[k]));
			}
			C[i][j] = _mm512_reduce_add_ps(add);
		}
	}
}


template<typename T>
void times_AT_B(MAT<T>& A, MAT<T>& B, MAT<T>& C) {
	int i, j, k = C.rows * C.cols;
	for (i = 0; i < k; ++i)
		C.data[i] = 0;
	for (k = 0; k < A.rows; ++k) {
		for (i = 0; i < C.rows; ++i) {
			for (j = 0; j < C.cols; ++j) {
				C[i][j] += A[k][i] * B[k][j];
			}
		}
	}
}

void times_AT_B(MAT<double>& A, MAT<double>& B, MAT<double>& C) {
	int i, j, k;
	ZeroMemory(C.data, sizeof(double) * C.rows * C.cols);
	int n = C.cols / 8;
	int m = C.cols - n * 8;
	__m512d mul;
	__m512d* tempB;
	__m512d* tempC;

	__mmask8 mask = (1 << m) - 1;

	for (k = 0; k < A.rows; ++k) {
		tempB = (__m512d*)(B[k] + m);
		for (i = 0; i < C.rows; ++i) {
			tempC = (__m512d*)(C[i] + m);
			mul = _mm512_set1_pd(A[k][i]);
			*(__m512d*)(C[i]) = _mm512_add_pd(*(__m512d*)(C[i]), _mm512_mask_blend_pd(mask, _mm512_setzero_pd(), _mm512_mul_pd(mul, *(__m512d*)(B[k]))));
			for (j = 0; j < n; ++j) {
				tempC[j] = _mm512_add_pd(tempC[j], _mm512_mul_pd(mul, tempB[j]));
			}
		}
	}
}
void times_AT_B(MAT<float>& A, MAT<float>& B, MAT<float>& C) {
	int i, j, k;
	ZeroMemory(C.data, sizeof(float) * C.rows * C.cols);
	int n = B.cols / 16;
	int m = B.cols - n * 16;
	__m512 mul;
	__m512* tempB;
	__m512* tempC;

	__mmask16 mask = (1 << m) - 1;

	for (k = 0; k < A.rows; ++k) {
		tempB = (__m512*)(B[k] + m);
		for (i = 0; i < C.rows; ++i) {
			tempC = (__m512*)(C[i] + m);
			mul = _mm512_set1_ps(A[k][i]);
			*(__m512*)(C[i]) = _mm512_add_ps(*(__m512*)(C[i]), _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), _mm512_mul_ps(mul, *(__m512*)(B[k]))));
			for (j = 0; j < n; ++j) {
				tempC[j] = _mm512_add_ps(tempC[j], _mm512_mul_ps(mul, tempB[j]));
			}
		}
	}
}

//===================================================================================

template<typename T>
void times(T* a, T b, T* c, long long len) {
	for (long long i = 0; i < len; ++i) {
		c[i] = a[i] * b;
	}
}
void times(double* a, double b, double* c, long long len) {
	int n = len / 8;
	int m = len - n * 8;
	__m512d mul = _mm512_set1_pd(b);
	__m512d* tempA = (__m512d*)(a + m);
	__m512d* tempC = (__m512d*)(c + m);

	*(__m512d*)c = _mm512_mask_blend_pd((1 << m) - 1, _mm512_setzero_pd(), _mm512_mul_pd(mul, *(__m512d*)a) );

	for (int i = 0; i < n; ++i) {
		tempC[i] = _mm512_mul_pd(mul, tempA[i]);
	}
}
void times(float* a, float b, float* c, long long len) {
	int n = len / 16;
	int m = len - n * 16;
	__m512 mul = _mm512_set1_ps(b);
	__m512* tempA = (__m512*)(a + m);
	__m512* tempC = (__m512*)(c + m);

	*(__m512*)c = _mm512_mask_blend_ps((1 << m) - 1, _mm512_setzero_ps(), _mm512_mul_ps(mul, *(__m512*)a));

	for (int i = 0; i < n; ++i) {
		tempC[i] = _mm512_mul_ps(mul, tempA[i]);
	}
}

template<typename T>
void add(T* a, T* b, T* c, long long len) {
	for (long long i = 0; i < len; ++i) {
		c[i] = a[i] + b[i];
	}
}
void add(double* a, double* b, double* c, long long len) {
	int n = len / 8;
	int m = len - n * 8;
	__m512d* tempA = (__m512d*)(a + m);
	__m512d* tempB = (__m512d*)(b + m);
	__m512d* tempC = (__m512d*)(c + m);

	*(__m512d*)c = _mm512_mask_blend_pd((1 << m) - 1, _mm512_setzero_pd(), _mm512_add_pd( *(__m512d*)a, *(__m512d*)b) );

	for (int i = 0; i < n; ++i) {
		tempC[i] = _mm512_add_pd(tempA[i], tempB[i]);
	}
}
void add(float* a, float* b, float* c, long long len) {
	int n = len / 16;
	int m = len - n * 16;
	__m512* tempA = (__m512*)(a + m);
	__m512* tempB = (__m512*)(b + m);
	__m512* tempC = (__m512*)(c + m);

	*(__m512*)c = _mm512_mask_blend_ps((1 << m) - 1, _mm512_setzero_ps(), _mm512_add_ps(*(__m512*)a, *(__m512*)b));

	for (int i = 0; i < n; ++i) {
		tempC[i] = _mm512_add_ps(tempA[i], tempB[i]);
	}
}

template<typename T>
void sub(T* a, T* b, T* c, long long len) {
	for (long long i = 0; i < len; ++i) {
		c[i] = a[i] - b[i];
	}
}
void sub(double* a, double* b, double* c, long long len) {
	int n = len / 8;
	int m = len - n * 8;
	__m512d* tempA = (__m512d*)(a + m);
	__m512d* tempB = (__m512d*)(b + m);
	__m512d* tempC = (__m512d*)(c + m);

	*(__m512d*)c = _mm512_mask_blend_pd((1 << m) - 1, _mm512_setzero_pd(), _mm512_sub_pd(*(__m512d*)a, *(__m512d*)b));

	for (int i = 0; i < n; ++i) {
		tempC[i] = _mm512_sub_pd(tempA[i], tempB[i]);
	}
}
void sub(float* a, float* b, float* c, long long len) {
	int n = len / 16;
	int m = len - n * 16;
	__m512* tempA = (__m512*)(a + m);
	__m512* tempB = (__m512*)(b + m);
	__m512* tempC = (__m512*)(c + m);

	*(__m512*)c = _mm512_mask_blend_ps((1 << m) - 1, _mm512_setzero_ps(), _mm512_sub_ps(*(__m512*)a, *(__m512*)b));

	for (int i = 0; i < n; ++i) {
		tempC[i] = _mm512_sub_ps(tempA[i], tempB[i]);
	}
}


template<class t>
MAT<t> operator * (t a, const MAT<t>& A) {
	MAT<t> n(A.rows, A.cols);

	times(A.data, a, n.data, A.rows * A.cols);

	return n;
}
template<class t>
MAT<t> operator * (const MAT<t>& A, t a) {
	MAT<t> n(A.rows, A.cols);

	times(A.data, a, n.data, A.rows * A.cols);

	return n;
}

template<class t>
MAT<t> operator * (const MAT<t>& A, const MAT<t>& B) {
	MAT<t> n(A.rows, B.cols);

	times_A_B(A, B, n);

	return n;
}
template<class t>
MAT<t> operator + (const MAT<t>& A, const MAT<t>& B) {
	MAT<t> n(A.rows, A.cols);

	add(A.data, B.data, n.data, A.rows * A.cols);

	return n;
}
template<class t>
MAT<t> operator - (const MAT<t>& A, const MAT<t>& B) {
	MAT<t> n(A.rows, A.cols);

	sub(A.data, B.data, n.data, A.rows * A.cols);

	return n;
}