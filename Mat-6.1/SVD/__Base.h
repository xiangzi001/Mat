
template<typename T>
void __SVD_HouseHolder_left(MAT<T>& A, MAT<T>& U, T* w, T* v, int r, int len) {
	int i, j;
	T sum(0), L;
	for (i = len; i < A.rows; ++i){
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

	for (i = 0; i < U.rows; ++i)
	{
		sum = 0;
		for (j = len; j < U.rows; ++j)
			sum += U[i][j] * w[j];
		for (j = len; j < U.cols; ++j){
			U[i][j] = U[i][j] - w[j] * sum;
		}
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
void __SVD_HouseHolder_right(MAT<T>& A, MAT<T>& V, T* w, int r, int len) {
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
	for (i = 0; i < V.rows; ++i)
	{
		sum = 0;
		for (j = len; j < V.cols; ++j)
			sum += V[i][j] * w[j];
		for (j = len; j < V.cols; ++j){
			V[i][j] = V[i][j] - w[j] * sum;
		}
	}
}

///*
void __SVD_HouseHolder_left(MAT<double>& A, MAT<double>& U, double* w, double* v, int r, int len) {
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

	for (i = 0; i < U.rows; ++i) {
		temp2 = (__m512d*)(U[i] + len);
		sum = _mm512_setzero_pd();
		for (j = 0; j < n; ++j)
			sum = _mm512_add_pd(sum, _mm512_mul_pd(temp1[j], temp2[j]));
		sum = _mm512_set1_pd(_mm512_reduce_add_pd(sum));
		for (j = 0; j < n; ++j) {
			temp2[j] = _mm512_sub_pd(temp2[j], _mm512_mul_pd(sum, temp1[j]));
		}
	}

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

void __SVD_HouseHolder_right(MAT<double>& A, MAT<double>& V, double* w, int r, int len) {
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

	for (i = 0; i < V.rows; ++i) {
		temp2 = (__m512d*)(V[i] + len);
		sum = _mm512_setzero_pd();
		for (j = 0; j < n; ++j)
			sum = _mm512_add_pd(sum, _mm512_mul_pd(temp1[j], temp2[j]));
		sum = _mm512_set1_pd(_mm512_reduce_add_pd(sum));
		for (j = 0; j < n; ++j) {
			temp2[j] = _mm512_sub_pd(temp2[j], _mm512_mul_pd(sum, temp1[j]));
		}
	}
}
//*/

template<typename T>
void __SVD_Q_times(MAT<T>& A, T* w, T* a) {
	int i, j;
	for (i = 0; i < A.cols; ++i) {
		w[i] = 1.0 / a[i];
	}
	for (i = 0; i < A.rows; ++i) {
		for (j = 0; j < A.cols; ++j) {
			A[i][j] = A[i][j] * w[j];
		}
	}
}

void __SVD_Q_times(MAT<double>& A, double* w, double* a) {
	int i, j;
	int n = (A.cols - 1) / 8 + 1;
	__m512d sum = _mm512_set1_pd(1);
	__m512d* temp1 = (__m512d*)w;
	__m512d* temp2 = (__m512d*)a;

	for (i = 0; i < n; ++i) {
		temp1[i] = _mm512_div_pd(sum, temp2[i]);
	}
	for (i = A.cols; i < 8 * n; ++i) {
		w[i] = 1;
	}

	for (i = 0; i < A.rows; ++i) {
		temp2 = (__m512d*)(A[i]);
		for (j = 0; j < n; ++j) {
			temp2[j] = _mm512_mul_pd(temp2[j], temp1[j]);
		}
	}
}