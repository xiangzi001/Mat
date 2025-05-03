
template <typename T>
void LLT(MAT<T>& A, MAT<T>& L) {
	T sum;
	int i, j, k = L.rows * L.cols;
	for (i = 0; i < k; ++i) {
		L.data[i] = 0;
	}
	for (j = 0; j < A.cols; ++j) {
		sum = 0;
		for (k = 0; k < j; ++k) {
			sum = sum + L[j][k] * L[j][k];
		}
		L[j][j] = sqrt(A[j][j] - sum);

		for (i = j + 1; i < A.rows; ++i) {
			sum = 0;
			for (k = 0; k < j; ++k) {
				sum = sum + L[i][k] * L[j][k];
			}
			L[i][j] = (A[i][j] - sum) / L[j][j];
		}
	}
}

void LLT(MAT<double>& A, MAT<double>& L) {
	int s = (int)(A.cols / 8) * 8;
	int m = A.cols - s, n;

	__m512d sum;
	__m512d* tempi;
	__m512d* tempj;
	__mmask8 mask = (1 << m) - 1;

	int i, j, k;
	ZeroMemory(L.data, sizeof(double) * L.rows * L.cols);

	for (j = 0; j < s; ++j) {
		n = 1 + (j - 1) / 8;
		tempj = (__m512d*)(L[j]);

		sum = _mm512_setzero_pd();
		for (k = 0; k < n; ++k) {
			sum = _mm512_add_pd(sum, _mm512_mul_pd(tempj[k], tempj[k]));
		}
		L[j][j] = sqrt(A[j][j] - _mm512_reduce_add_pd(sum));

		for (i = j + 1; i < A.rows; ++i) {
			tempi = (__m512d*)(L[i]);
			sum = _mm512_setzero_pd();
			for (k = 0; k < n; ++k) {
				sum = _mm512_add_pd(sum, _mm512_mul_pd(tempi[k], tempj[k]));
			}
			L[i][j] = (A[i][j] - _mm512_reduce_add_pd(sum)) / L[j][j];
		}
	}
	n = A.cols / 8;
	for (j = s; j < A.cols; ++j) {
		tempj = (__m512d*)(L[j] + m);
		sum = _mm512_mask_blend_pd(mask, _mm512_setzero_pd(), _mm512_mul_pd(*(__m512d*)(L[j]), *(__m512d*)(L[j])));
		for (k = 0; k < n; ++k) {
			sum = _mm512_add_pd(sum, _mm512_mul_pd(tempj[k], tempj[k]));
		}
		L[j][j] = sqrt(A[j][j] - _mm512_reduce_add_pd(sum));

		for (i = j + 1; i < A.rows; ++i) {
			tempi = (__m512d*)(L[i] + m);
			sum = _mm512_mask_blend_pd(mask, _mm512_setzero_pd(), _mm512_mul_pd(*(__m512d*)(L[i]), *(__m512d*)(L[j])));
			for (k = 0; k < n; ++k) {
				sum = _mm512_add_pd(sum, _mm512_mul_pd(tempi[k], tempj[k]));
			}
			L[i][j] = (A[i][j] - _mm512_reduce_add_pd(sum)) / L[j][j];
		}
	}
}

void LLT(MAT<float>& A, MAT<float>& L) {
	int s = (int)(A.cols / 16) * 16;
	int m = A.cols - s, n;

	__m512 sum;
	__m512* tempi;
	__m512* tempj;
	__mmask16 mask = (1 << m) - 1;

	int i, j, k;
	ZeroMemory(L.data, sizeof(float) * L.rows * L.cols);

	for (j = 0; j < s; ++j) {
		n = 1 + (j - 1) / 16;
		tempj = (__m512*)(L[j]);

		sum = _mm512_setzero_ps();
		for (k = 0; k < n; ++k) {
			sum = _mm512_add_ps(sum, _mm512_mul_ps(tempj[k], tempj[k]));
		}
		L[j][j] = sqrt(A[j][j] - _mm512_reduce_add_ps(sum));

		for (i = j + 1; i < A.rows; ++i) {
			tempi = (__m512*)(L[i]);
			sum = _mm512_setzero_ps();
			for (k = 0; k < n; ++k) {
				sum = _mm512_add_ps(sum, _mm512_mul_ps(tempi[k], tempj[k]));
			}
			L[i][j] = (A[i][j] - _mm512_reduce_add_ps(sum)) / L[j][j];
		}
	}
	n = A.cols / 16;
	for (j = s; j < A.cols; ++j) {
		tempj = (__m512*)(L[j] + m);
		sum = _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), _mm512_mul_ps(*(__m512*)(L[j]), *(__m512*)(L[j])));
		for (k = 0; k < n; ++k) {
			sum = _mm512_add_ps(sum, _mm512_mul_ps(tempj[k], tempj[k]));
		}
		L[j][j] = sqrt(A[j][j] - _mm512_reduce_add_ps(sum));

		for (i = j + 1; i < A.rows; ++i) {
			tempi = (__m512*)(L[i] + m);
			sum = _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), _mm512_mul_ps(*(__m512*)(L[i]), *(__m512*)(L[j])));
			for (k = 0; k < n; ++k) {
				sum = _mm512_add_ps(sum, _mm512_mul_ps(tempi[k], tempj[k]));
			}
			L[i][j] = (A[i][j] - _mm512_reduce_add_ps(sum)) / L[j][j];
		}
	}
}

template <typename T>
void LDLT(MAT<T>& A, MAT<T>& L, MAT<T>& D) {
	int i, j, k = L.rows * L.cols;
	T temp;
	for (i = 0; i < k; ++i) {
		L.data[i] = 0;
	}
	for (i = 0; i < k; ++i) {
		D.data[i] = 0;
	}
	for (i = 0; i < A.rows; ++i)
	{
		temp = 0;
		for (j = 0; j < i; ++j) {
			temp = temp + L[i][j] * L[i][j] * D[j][j];
		}
		D[i][i] = A[i][i] - temp;

		L[i][i] = 1;
		for (j = i + 1; j < L.rows; ++j) {
			temp = 0;
			for (k = 0; k < i; ++k) {
				temp = temp + L[i][k] * L[j][k] * D[k][k];
			}
			L[j][i] = (A[j][i] - temp) / D[i][i];
		}
	}
}

void LDLT(MAT<double>& A, MAT<double>& L, MAT<double>& D) {
	int i, j, k;
	int	n = A.rows / 8;
	const int m = A.rows - n * 8, s = n * 8;
	double* d = D.data;
	ZeroMemory(L.data, sizeof(double) * L.rows * L.cols);
	ZeroMemory(D.data, sizeof(double) * D.rows * D.cols);

	__m512d sum;
	__m512d* tempLi;
	__m512d* tempLj;
	__m512d* tempD = (__m512d*)d;
	__mmask8 mask = (1 << m) - 1;

	for (i = 0; i < s; ++i){
		n = (i + 7) / 8;
		tempLi = (__m512d*)(L[i]);

		sum = _mm512_mul_pd(_mm512_mul_pd(tempLi[0], tempLi[0]), tempD[0]);
		for (j = 1; j < n; ++j) {
			sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_mul_pd(tempLi[j], tempLi[j]), tempD[j]));
		}
		d[i] = A[i][i] - _mm512_reduce_add_pd(sum);

		L[i][i] = 1;
		for (j = i + 1; j < L.rows; ++j) {
			tempLj = (__m512d*)(L[j]);
			sum = _mm512_mul_pd(_mm512_mul_pd(tempLi[0], tempLj[0]), tempD[0]);
			for (k = 1; k < n; ++k) {
				sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_mul_pd(tempLi[k], tempLj[k]), tempD[k]));
			}
			L[j][i] = (A[j][i] - _mm512_reduce_add_pd(sum)) / d[i];
		}
	}
	n = A.cols / 8;
	for (i = s; i < A.cols; ++i) {
		tempLi = (__m512d*)(L[i] + m);
		tempD = (__m512d*)(d + m);

		sum = _mm512_mask_blend_pd(mask, _mm512_setzero_pd(), _mm512_mul_pd(_mm512_mul_pd(*(__m512d*)(L[i]), *(__m512d*)(L[i])), *(__m512d*)(d)));
		for (j = 0; j < n; ++j) {
			sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_mul_pd(tempLi[j], tempLi[j]), tempD[j]));
		}
		d[i] = A[i][i] - _mm512_reduce_add_pd(sum);

		L[i][i] = 1;
		for (j = i + 1; j < L.rows; ++j) {
			sum = _mm512_mask_blend_pd(mask, _mm512_setzero_pd(), _mm512_mul_pd(_mm512_mul_pd(*(__m512d*)(L[i]), *(__m512d*)(L[j])), *(__m512d*)(d)));
			for (k = 0; k < n; ++k) {
				tempLj = (__m512d*)(L[j] + m);
				sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_mul_pd(tempLi[k], tempLj[k]), tempD[k]));
			}
			L[j][i] = (A[j][i] - _mm512_reduce_add_pd(sum)) / d[i];
		}
	}
	for (i = 1; i < D.rows; ++i) {
		D[i][i] = d[i];
		d[i] = 0;
	}
}

void LDLT(MAT<float>& A, MAT<float>& L, MAT<float>& D) {
	int i, j, k;
	int	n = A.rows / 16;
	const int m = A.rows - n * 16, s = n * 16;
	float* d = D.data;
	ZeroMemory(L.data, sizeof(float) * L.rows * L.cols);
	ZeroMemory(D.data, sizeof(float) * D.rows * D.cols);

	__m512 sum;
	__m512* tempLi;
	__m512* tempLj;
	__m512* tempD = (__m512*)d;
	__mmask16 mask = (1 << m) - 1;

	for (i = 0; i < s; ++i) {
		n = (i + 15) / 16;
		tempLi = (__m512*)(L[i]);

		sum = _mm512_mul_ps(_mm512_mul_ps(tempLi[0], tempLi[0]), tempD[0]);
		for (j = 1; j < n; ++j) {
			sum = _mm512_add_ps(sum, _mm512_mul_ps(_mm512_mul_ps(tempLi[j], tempLi[j]), tempD[j]));
		}
		d[i] = A[i][i] - _mm512_reduce_add_ps(sum);

		L[i][i] = 1;
		for (j = i + 1; j < L.rows; ++j) {
			tempLj = (__m512*)(L[j]);
			sum = _mm512_mul_ps(_mm512_mul_ps(tempLi[0], tempLj[0]), tempD[0]);
			for (k = 1; k < n; ++k) {
				sum = _mm512_add_ps(sum, _mm512_mul_ps(_mm512_mul_ps(tempLi[k], tempLj[k]), tempD[k]));
			}
			L[j][i] = (A[j][i] - _mm512_reduce_add_ps(sum)) / d[i];
		}
	}
	n = A.cols / 16;
	for (i = s; i < A.cols; ++i) {
		tempLi = (__m512*)(L[i] + m);
		tempD = (__m512*)(d + m);

		sum = _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), _mm512_mul_ps(_mm512_mul_ps(*(__m512*)(L[i]), *(__m512*)(L[i])), *(__m512*)(d)));
		for (j = 0; j < n; ++j) {
			sum = _mm512_add_ps(sum, _mm512_mul_ps(_mm512_mul_ps(tempLi[j], tempLi[j]), tempD[j]));
		}
		d[i] = A[i][i] - _mm512_reduce_add_ps(sum);

		L[i][i] = 1;
		for (j = i + 1; j < L.rows; ++j) {
			sum = _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), _mm512_mul_ps(_mm512_mul_ps(*(__m512*)(L[i]), *(__m512*)(L[j])), *(__m512*)(d)));
			for (k = 0; k < n; ++k) {
				tempLj = (__m512*)(L[j] + m);
				sum = _mm512_add_ps(sum, _mm512_mul_ps(_mm512_mul_ps(tempLi[k], tempLj[k]), tempD[k]));
			}
			L[j][i] = (A[j][i] - _mm512_reduce_add_ps(sum)) / d[i];
		}
	}
	for (i = 1; i < D.rows; ++i) {
		D[i][i] = d[i];
		d[i] = 0;
	}
}
