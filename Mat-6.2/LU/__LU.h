

void __LU(MAT<double>& L, MAT<double>& U) {
	int x, y, j;
	int s = U.cols % 8;
	double t;
	int n1 = U.cols / 8;
	int m1 = s;
	const __mmask8 mask1 = (1 << m1) - 1;

	__m512d temp;
	__m512d* temp1;
	__m512d* temp2;

	for (y = 0; y < s; ++y) {
		t = 1.0 / U[y][y];
		temp2 = (__m512d*)(U[y] + m1);
		for (x = y + 1; x < U.rows; ++x) {
			L[x][y] = U[x][y] * t;
			temp = _mm512_set1_pd(-L[x][y]);
			*(__m512d*)(U[x]) = _mm512_add_pd(*(__m512d*)(U[x]), _mm512_mul_pd(_mm512_mask_blend_pd(mask1, _mm512_setzero_pd(), temp), *(__m512d*)(U[y])));
			temp1 = (__m512d*)(U[x] + m1);
			for (j = 0; j < n1; ++j)
				temp1[j] = _mm512_add_pd(temp1[j], _mm512_mul_pd(temp, temp2[j]));
		}
	}
	for (y = s; y < U.cols - 1; ++y) {
		n1 = (U.cols - y - 1) / 8 + 1;
		m1 = U.cols - 8 * n1;
		t = 1.0 / U[y][y];
		temp2 = (__m512d*)(U[y] + m1);
		for (x = y + 1; x < U.rows; ++x) {
			L[x][y] = U[x][y] * t;
			temp = _mm512_set1_pd(-L[x][y]);
			temp1 = (__m512d*)(U[x] + m1);
			for (j = 0; j < n1; ++j)
				temp1[j] = _mm512_add_pd(temp1[j], _mm512_mul_pd(temp, temp2[j]));
		}
	}
}

void __LU(MAT<float>& L, MAT<float>& U) {
	int x, y, j;
	int s = U.cols % 16;
	float t;
	int n1 = U.cols / 16;
	int m1 = s;
	const __mmask16 mask1 = (1 << m1) - 1;

	__m512 temp;
	__m512* temp1;
	__m512* temp2;

	for (y = 0; y < s; ++y) {
		t = 1.0 / U[y][y];
		temp2 = (__m512*)(U[y] + m1);
		for (x = y + 1; x < U.rows; ++x) {
			L[x][y] = U[x][y] * t;
			temp = _mm512_set1_ps(-L[x][y]);
			*(__m512*)(U[x]) = _mm512_add_ps(*(__m512*)(U[x]), _mm512_mul_ps(_mm512_mask_blend_ps(mask1, _mm512_setzero_ps(), temp), *(__m512*)(U[y])));
			temp1 = (__m512*)(U[x] + m1);
			for (j = 0; j < n1; ++j)
				temp1[j] = _mm512_add_ps(temp1[j], _mm512_mul_ps(temp, temp2[j]));
		}
	}
	for (y = s; y < U.cols - 1; ++y) {
		n1 = (U.cols - y - 1) / 16 + 1;
		m1 = U.cols - 16 * n1;
		t = 1.0 / U[y][y];
		temp2 = (__m512*)(U[y] + m1);
		for (x = y + 1; x < U.rows; ++x) {
			L[x][y] = U[x][y] * t;
			temp = _mm512_set1_ps(-L[x][y]);
			temp1 = (__m512*)(U[x] + m1);
			for (j = 0; j < n1; ++j)
				temp1[j] = _mm512_add_ps(temp1[j], _mm512_mul_ps(temp, temp2[j]));
		}
	}
}

template <typename T>
void LU(MAT<T>& A, MAT<T>& L, MAT<T>& U)
{
	int i, j, k = U.rows * U.cols;

	for (i = 0; i < k; ++i) {
		U.data[i] = 0;
	}
	for (i = 0; i < k; ++i) {
		L.data[i] = 0;
	}
	for (i = 0; i < L.rows; ++i) {
		L[i][i] = 1;
	}

	for (i = 0; i < L.rows; ++i) {
		for (j = 0; j < i; ++j) {
			L[i][j] = A[i][j];
			for (k = 0; k < j; ++k)
			{
				L[i][j] -= L[i][k] * U[k][j];
			}
			L[i][j] /= U[j][j];

		}

		for (j = i; j < L.rows; ++j) {
			U[i][j] = A[i][j];
			for (k = 0; k < i; ++k) {
				U[i][j] -= L[i][k] * U[k][j];
			}
		}
	}
}

void LU(MAT<double>& A, MAT<double>& L, MAT<double>& U)
{
	int i, k = U.rows * U.cols;
	memcpy(U.data, A.data, sizeof(double) * k);
	ZeroMemory(L.data, sizeof(double) * k);
	for (i = 0; i < L.rows; ++i) {
		L[i][i] = 1;
	}
	__LU(L, U);
}

void LU(MAT<float>& A, MAT<float>& L, MAT<float>& U)
{
	int i, k = U.rows * U.cols;
	memcpy(U.data, A.data, sizeof(float) * k);
	ZeroMemory(L.data, sizeof(float) * k);
	for (i = 0; i < L.rows; ++i) {
		L[i][i] = 1;
	}
	__LU(L, U);
}