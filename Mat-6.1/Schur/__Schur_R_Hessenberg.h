
template<typename T>
void __Schur_HouseHolder_R(MAT<T>& Q, MAT<T>& R, T* w, T* v) {
	int i, j, k, r;

	T sum, cr, hr;

	for (r = 0; r < R.rows - 2; ++r) {

		for (i = r + 1; i < R.rows; ++i) {
			w[i] = R[i][r];
		}
		sum = 0;
		for (k = r + 1; k < R.rows; ++k)
			sum += w[k] * w[k];
		cr = sqrt(sum);

		if (w[r + 1] < 0.0)cr = -cr;
		w[r + 1] = R[r + 1][r] + cr;
		hr = 1.0 / sqrt(cr * w[r + 1]);

		for (i = r + 1; i < R.rows; ++i) {
			w[i] *= hr;
		}

		for (i = 0; i < R.rows; ++i)//定义Q
		{
			sum = 0;
			for (j = r + 1; j < R.rows; ++j)
				sum += Q[i][j] * w[j];
			for (j = r + 1; j < R.cols; ++j) {
				Q[i][j] = Q[i][j] - w[j] * sum;
			}
		}
		for (i = 0; i < R.rows; ++i)//定义R
		{
			sum = 0;
			for (j = r + 1; j < R.rows; ++j)
				sum += R[i][j] * w[j];
			for (j = r + 1; j < R.cols; ++j) {
				R[i][j] = R[i][j] - w[j] * sum;
			}
		}
		for (j = 0; j < R.cols; ++j) {
			v[j] = 0;
		}
		for (i = r + 1; i < R.rows; ++i) {
			for (j = r; j < R.cols; ++j) {
				v[j] += w[i] * R[i][j];
			}
		}
		for (i = r + 1; i < R.rows; ++i) {
			for (j = r; j < R.cols; ++j) {
				R[i][j] -= w[i] * v[j];
			}
		}
	}
}


void __Schur_HouseHolder_R(MAT<double>& Q, MAT<double>& R, double* w, double* v) {

	int i, j, k, r;
	int n;
	double L;

	__m512d sum;
	__m512d* temp1;
	__m512d* temp2;

	ZeroMemory(w, sizeof(double) * (R.rows + 7));
	ZeroMemory(v, sizeof(double) * (R.cols + 7));

	for (r = 0; r < R.rows - 2; ++r) {
		n = (R.rows - r - 1) / 8 + 1;
		temp1 = (__m512d*)(w + r);

		for (i = r + 1; i < R.rows; ++i) {
			w[i] = R[i][r];
		}

		sum = _mm512_setzero_pd();
		for (k = 0; k < n; ++k)
			sum = _mm512_add_pd(sum, _mm512_mul_pd(temp1[k], temp1[k]));
		L = sqrt(_mm512_reduce_add_pd(sum));

		if (w[r + 1] < 0.0)L = -L;
		w[r + 1] += L;
		sum = _mm512_set1_pd(1.0 / sqrt(L * w[r + 1]));

		for (k = 0; k < n; ++k)
			temp1[k] = _mm512_mul_pd(sum, temp1[k]);

		for (i = 0; i < Q.rows; ++i) {
			temp2 = (__m512d*)(Q[i] + r);
			sum = _mm512_setzero_pd();
			for (j = 0; j < n; ++j)
				sum = _mm512_add_pd(sum, _mm512_mul_pd(temp1[j], temp2[j]));
			sum = _mm512_set1_pd(_mm512_reduce_add_pd(sum));

			for (j = 0; j < n; ++j) {
				temp2[j] = _mm512_sub_pd(temp2[j], _mm512_mul_pd(sum, temp1[j]));
			}
		}

		for (i = 0; i < R.rows; ++i) {
			temp2 = (__m512d*)(R[i] + r);
			sum = _mm512_setzero_pd();
			for (j = 0; j < n; ++j)
				sum = _mm512_add_pd(sum, _mm512_mul_pd(temp1[j], temp2[j]));
			sum = _mm512_set1_pd(_mm512_reduce_add_pd(sum));

			for (j = 0; j < n; ++j) {
				temp2[j] = _mm512_sub_pd(temp2[j], _mm512_mul_pd(sum, temp1[j]));
			}
		}

		ZeroMemory(v, sizeof(double) * R.cols);
		n = (R.cols - r - 1) / 8 + 1;
		temp1 = (__m512d*)(v + r);

		for (i = r + 1; i < R.rows; ++i) {
			temp2 = (__m512d*)(R[i] + r);
			sum = _mm512_set1_pd(w[i]);
			for (j = 0; j < n; ++j) {
				temp1[j] = _mm512_add_pd(temp1[j], _mm512_mul_pd(temp2[j], sum));
			}
		}
		ZeroMemory(v + R.cols, sizeof(double) * 7);
		for (i = r + 1; i < R.rows; ++i) {
			temp2 = (__m512d*)(R[i] + r);
			sum = _mm512_set1_pd(w[i]);
			for (j = 0; j < n; ++j) {
				temp2[j] = _mm512_sub_pd(temp2[j], _mm512_mul_pd(temp1[j], sum));
			}
		}
		w[r + 1] = 0;
	}
}

void __Schur_HouseHolder_R(MAT<float>& Q, MAT<float>& R, float* w, float* v) {

	int i, j, k, r;
	int n;
	float L;

	__m512 sum;
	__m512* temp1;
	__m512* temp2;

	ZeroMemory(w, sizeof(float) * (R.rows + 15));
	ZeroMemory(v, sizeof(float) * (R.cols + 15));

	for (r = 0; r < R.rows - 2; ++r) {
		n = (R.rows - r - 1) / 16 + 1;
		temp1 = (__m512*)(w + r);

		for (i = r + 1; i < R.rows; ++i) {
			w[i] = R[i][r];
		}

		sum = _mm512_setzero_ps();
		for (k = 0; k < n; ++k)
			sum = _mm512_add_ps(sum, _mm512_mul_ps(temp1[k], temp1[k]));
		L = sqrt(_mm512_reduce_add_ps(sum));

		if (w[r + 1] < 0.0)L = -L;
		w[r + 1] += L;
		sum = _mm512_set1_ps(1.0 / sqrt(L * w[r + 1]));

		for (k = 0; k < n; ++k)
			temp1[k] = _mm512_mul_ps(sum, temp1[k]);

		for (i = 0; i < Q.rows; ++i) {
			temp2 = (__m512*)(Q[i] + r);
			sum = _mm512_setzero_ps();
			for (j = 0; j < n; ++j)
				sum = _mm512_add_ps(sum, _mm512_mul_ps(temp1[j], temp2[j]));
			sum = _mm512_set1_ps(_mm512_reduce_add_ps(sum));

			for (j = 0; j < n; ++j) {
				temp2[j] = _mm512_sub_ps(temp2[j], _mm512_mul_ps(sum, temp1[j]));
			}
		}

		for (i = 0; i < R.rows; ++i) {
			temp2 = (__m512*)(R[i] + r);
			sum = _mm512_setzero_ps();
			for (j = 0; j < n; ++j)
				sum = _mm512_add_ps(sum, _mm512_mul_ps(temp1[j], temp2[j]));
			sum = _mm512_set1_ps(_mm512_reduce_add_ps(sum));

			for (j = 0; j < n; ++j) {
				temp2[j] = _mm512_sub_ps(temp2[j], _mm512_mul_ps(sum, temp1[j]));
			}
		}

		ZeroMemory(v, sizeof(float) * R.cols);
		n = (R.cols - r - 1) / 16 + 1;
		temp1 = (__m512*)(v + r);

		for (i = r + 1; i < R.rows; ++i) {
			temp2 = (__m512*)(R[i] + r);
			sum = _mm512_set1_ps(w[i]);
			for (j = 0; j < n; ++j) {
				temp1[j] = _mm512_add_ps(temp1[j], _mm512_mul_ps(temp2[j], sum));
			}
		}
		ZeroMemory(v + R.cols, sizeof(float) * 15);
		for (i = r + 1; i < R.rows; ++i) {
			temp2 = (__m512*)(R[i] + r);
			sum = _mm512_set1_ps(w[i]);
			for (j = 0; j < n; ++j) {
				temp2[j] = _mm512_sub_ps(temp2[j], _mm512_mul_ps(temp1[j], sum));
			}
		}
		w[r + 1] = 0;
	}
}

