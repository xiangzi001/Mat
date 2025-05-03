
template<typename T>
void __SV_HouseHolder_left(MAT<T>& A, T* w, T* v, int r, int len) {
	int i, j;
	T sum(0), L;
	for (i = len; i < A.rows; ++i) {
		w[i] = A[i][r];
	}
	for (i = len; i < A.rows; ++i)
		sum += w[i] * w[i];
	L = sqrt(sum);
	if (w[len] < 0.0)L = -L;
	w[len] += L;

	L = 1 / sqrt(L * w[len]);
	for (i = len; i < A.rows; ++i) {
		w[i] *= L;
	}

	for (j = 0; j < A.cols; ++j) {
		v[j] = 0;
	}
	for (i = len; i < A.rows; ++i) {
		for (j = 0; j < A.cols; ++j) {
			v[j] += w[i] * A[i][j];
		}
	}
	for (i = len; i < A.rows; ++i) {
		for (j = 0; j < A.cols; ++j) {
			A[i][j] -= w[i] * v[j];
		}
	}
}
template<typename T>
void __SV_HouseHolder_right(MAT<T>& A, T*& w, int r, int len) {
	int i, j;
	T sum(0), L;

	for (i = len; i < A.cols; ++i) {
		w[i] = A[r][i];
	}
	for (i = len; i < A.cols; ++i)
		sum += w[i] * w[i];
	L = sqrt(sum);
	if (w[len] < 0.0)L = -L;
	w[len] += L;

	L = 1 / sqrt(L * w[len]);
	for (i = len; i < A.cols; ++i) {
		w[i] *= L;
	}

	for (j = r; j < A.rows; ++j)
	{
		sum = 0;
		for (i = len; i < A.cols; ++i)
			sum += A[j][i] * w[i];
		for (i = len; i < A.cols; ++i)
			A[j][i] = A[j][i] - w[i] * sum;
	}
}

void __SV_HouseHolder_left(MAT<double>& A, double* w, double* v, int r, int len) {
	int i, j;

	w = w - len;

	__m512d sum = _mm512_setzero_pd();
	__m512d* temp1 = (__m512d*)(w + len), * temp2;
	int n = (A.rows - len - 1) / 8 + 1;
	double L;

	for (i = len; i < A.rows; ++i) {
		w[i] = A[i][r];
	}
	ZeroMemory(w + A.rows, sizeof(double)* (len + n * 8 - A.rows));
	for (i = 0; i < n; ++i)
		sum = _mm512_add_pd(sum, _mm512_mul_pd(temp1[i], temp1[i]));
	L = sqrt(_mm512_reduce_add_pd(sum));

	if (w[len] < 0.0)L = -L;
	w[len] += L;

	sum = _mm512_set1_pd(1.0 / sqrt(L * w[len]));
	for (i = 0; i < n; ++i)
		temp1[i] = _mm512_mul_pd(sum, temp1[i]);

	ZeroMemory(v, sizeof(double) * (A.cols - r));
	n = (A.cols - r - 1) / 8 + 1;
	temp1 = (__m512d*)(v);

	for (i = len; i < A.rows; ++i) {
		temp2 = (__m512d*)(A[i] + r);
		sum = _mm512_set1_pd(w[i]);
		for (j = 0; j < n; ++j) {
			temp1[j] = _mm512_add_pd(temp1[j], _mm512_mul_pd(temp2[j], sum));
		}
	}
	ZeroMemory(v + A.cols - r, sizeof(double) * (8 * n - A.cols + r));
	for (i = len; i < A.rows; ++i) {
		temp2 = (__m512d*)(A[i] + r);
		sum = _mm512_set1_pd(w[i]);
		for (j = 0; j < n; ++j) {
			temp2[j] = _mm512_sub_pd(temp2[j], _mm512_mul_pd(temp1[j], sum));
		}
	}
}

void __SV_HouseHolder_right(MAT<double>& A, double* w, int r, int len) {
	int i, j;

	w = w - len;

	__m512d sum = _mm512_setzero_pd();
	__m512d* temp1 = (__m512d*)(w + len), * temp2;
	int n = (A.cols - len - 1) / 8 + 1;
	double L;

	for (i = len; i < A.cols; ++i) {
		w[i] = A[r][i];
	}
	ZeroMemory(w + A.cols, sizeof(double)* (len + n * 8 - A.cols));
	for (i = 0; i < n; ++i)
		sum = _mm512_add_pd(sum, _mm512_mul_pd(temp1[i], temp1[i]));
	L = sqrt(_mm512_reduce_add_pd(sum));

	if (w[len] < 0.0)L = -L;
	w[len] += L;

	sum = _mm512_set1_pd(1.0 / sqrt(L * w[len]));
	for (i = 0; i < n; ++i)
		temp1[i] = _mm512_mul_pd(sum, temp1[i]);


	for (i = r; i < A.rows; ++i) {
		temp2 = (__m512d*)(A[i] + len);
		sum = _mm512_setzero_pd();
		for (j = 0; j < n; ++j)
			sum = _mm512_add_pd(sum, _mm512_mul_pd(temp1[j], temp2[j]));
		sum = _mm512_set1_pd(_mm512_reduce_add_pd(sum));
		for (j = 0; j < n; ++j) {
			temp2[j] = _mm512_sub_pd(temp2[j], _mm512_mul_pd(sum, temp1[j]));
		}
	}
}


template<typename T>
void __SV_Givens_UP(T*& a, T*& b, int m, int n, T eps) {
	int i, k;
	T temp, t, s, c, B;

	while (n > m) {
		temp = a[m] * a[m] - a[n] * a[n] - b[n - 1] * b[n - 1];
		t = b[m] * a[m] / temp;
		//t = b[m] / a[m];
		if (fabs(t) < eps) {
			++m;
			continue;
		}
		c = 1 / sqrt(t * t + 1);
		s = t * c;

		temp = c * a[m] + s * b[m];
		b[m] = c * b[m] - s * a[m];
		a[m] = temp;
		B = s * a[m + 1];
		a[m + 1] = c * a[m + 1];

		t = B / a[m];
		c = 1 / sqrt(t * t + 1);
		s = t * c;

		a[m] = c * a[m] + s * B;
		temp = c * b[m] + s * a[m + 1];
		a[m + 1] = c * a[m + 1] - s * b[m];
		b[m] = temp;
		B = s * b[m + 1];
		b[m + 1] = c * b[m + 1];

		for (i = m + 1; i < n; ++i) {

			t = B / b[i - 1];
			if (fabs(t) < eps) {
				if (n > i + 1) {
					__SV_Givens_UP(a, b, m, i, eps);
					m = i + 1;
					break;
				}
				else {
					--n;
					break;
				}
			}
			c = 1 / sqrt(t * t + 1);
			s = t * c;

			b[i - 1] = c * b[i - 1] + s * B;
			temp = c * a[i] + s * b[i];
			b[i] = c * b[i] - s * a[i];
			a[i] = temp;
			B = s * a[i + 1];
			a[i + 1] = c * a[i + 1];

			t = B / a[i];
			c = 1 / sqrt(t * t + 1);
			s = t * c;

			a[i] = c * a[i] + s * B;
			temp = c * b[i] + s * a[i + 1];
			a[i + 1] = c * a[i + 1] - s * b[i];
			b[i] = temp;
			B = s * b[i + 1];
			b[i + 1] = c * b[i + 1];

		}
	}
}

template<typename T>
void __SV_QR(MAT<T>& A, MAT<T>& S, double eps) {
	int i = 0;
	int s = A.rows * A.cols;
	int r;
	for (i = 0; i < s; ++i) {
		S.data[i] = A.data[i];
	}

	T* w, * v;
	if (S.rows > S.cols) {
		//T* w = (T*)alloca(sizeof(T) * S.rows);
		s = sizeof(T) * (S.cols - 1) / 64 + 1;
		w = new T[s * 64 / sizeof(T)];
		v = new T[s * 64 / sizeof(T)];
		s = S.cols - 1;

		for (r = 0; r < s; ++r) {
			__SV_HouseHolder_left(S, w, v, r, r);
			__SV_HouseHolder_right(S, w, r, r + 1);
		}
		__SV_HouseHolder_left(S, w, v, s, s);

		T* a(S[S.rows - 2]);
		T* b(S[S.rows - 1]);
		for (r = 0; r < S.cols; ++r) {
			a[r] = S[r][r];
		}
		for (r = 1; r < S.cols; ++r) {
			b[r - 1] = S[r - 1][r];
		}
		__SV_Givens_UP(a, b, 0, S.cols - 1, eps);
		for (r = 0; r < S.cols; ++r) {
			S[r][r] = a[r];
		}
		for (r = 1; r < S.cols; ++r) {
			S[r - 1][r] = 0;
		}
		for (r = 0; r < S.cols; ++r) {
			a[r] = 0;
		}
	}
	else {
		//T* w = (T*)alloca(sizeof(T) * S.cols);
		s = sizeof(T) * (S.cols - 1) / 64 + 1;
		w = new T[s * 64 / sizeof(T)];
		v = new T[s * 64 / sizeof(T)];
		s = S.rows - 1;

		for (r = 0; r < s; ++r) {
			__SV_HouseHolder_right(S, w, r, r);
			__SV_HouseHolder_left(S, w, v, r, r + 1);
		}
		__SV_HouseHolder_right(S, w, s, s);

		T* a(S.data);
		T* b(a + S.rows);
		for (r = 0; r < S.rows; ++r) {
			a[r] = S[r][r];
		}
		for (r = 1; r < S.rows; ++r) {
			b[r - 1] = S[r][r - 1];
		}
		__SV_Givens_UP(a, b, 0, S.rows - 1, eps);
		for (r = 0; r < S.rows; ++r) {
			S[r][r] = a[r];
		}
		for (r = 1; r < S.rows; ++r) {
			S[r][r - 1] = 0;
		}
		for (r = 1; r < S.rows; ++r) {
			a[r] = 0;
		}
	}
	delete[] w;
	delete[] v;
}
