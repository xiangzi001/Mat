
template<typename T>
void __Schur_QR_R(MAT<T>& Q, MAT<T>& R, int m, int n, double eps) {
	int i, j, k, r;
	T sum, l, x, y, z;
	complex<T> shift;
	T q1[3];
	T q2[3];
	T q3[3];

	while (1) {

		if (fabs(R[m + 1][m]) < eps) {
			++m;
		}
		for (r = n - 1; r > m; --r) {
			if (fabs(R[r][r - 1]) < eps) {
				if (n - r <= 2) {
					n = r;
					break;
				}
				else {
					__Schur_QR_R(Q, R, r, n, eps);
					n = r;
					break;
				}
			}
		}

		x = R[n - 2][n - 2] + R[n - 1][n - 1];
		y = x * x - (R[n - 2][n - 2] * R[n - 1][n - 1] - R[n - 2][n - 1] * R[n - 1][n - 2]) * 4;
		if (n - m <= 2) {
			if (n - m < 2) {
				break;
			}
			else if (y < 0) {
				break;
			}
		}

		if (y < 0) {
			shift.real = x * 0.5;
			shift.imag = sqrt(-y) * 0.5;

			x = R[m][m] * R[m][m] + R[m][m + 1] * R[m + 1][m] - 2 * shift.real * R[m][m] + shift.len2();
			y = R[m + 1][m] * (R[m][m] + R[m + 1][m + 1] - 2 * shift.real);
			z = R[m + 1][m] * R[m + 2][m + 1];
			l = 1.0 / sqrt(x * x + y * y + z * z);
			q1[0] = x * l; q1[1] = y * l; q1[2] = z * l;

			q2[0] = -q1[0] * q1[1];		q2[1] = 1 - q1[1] * q1[1];	q2[2] = -q1[2] * q1[1];
			l = 1.0 / sqrt(q2[0] * q2[0] + q2[1] * q2[1] + q2[2] * q2[2]);
			q2[0] *= l; q2[1] *= l; q2[2] *= l;

			q3[0] = q1[1] * q2[2] - q1[2] * q2[1];
			q3[1] = q1[2] * q2[0] - q1[0] * q2[2];
			q3[2] = q1[0] * q2[1] - q1[1] * q2[0];

			for (i = 0; i < Q.rows; ++i) {
				x = Q[i][m] * q1[0] + Q[i][m + 1] * q1[1] + Q[i][m + 2] * q1[2];
				y = Q[i][m] * q2[0] + Q[i][m + 1] * q2[1] + Q[i][m + 2] * q2[2];
				Q[i][m + 2] = Q[i][m] * q3[0] + Q[i][m + 1] * q3[1] + Q[i][m + 2] * q3[2];
				Q[i][m + 1] = y;
				Q[i][m] = x;
			}

			if (m + 4 < n) k = m + 4;
			else k = n;
			for (i = m; i < R.cols; ++i) {
				x = R[m][i] * q1[0] + R[m + 1][i] * q1[1] + R[m + 2][i] * q1[2];
				y = R[m][i] * q2[0] + R[m + 1][i] * q2[1] + R[m + 2][i] * q2[2];
				R[m + 2][i] = R[m][i] * q3[0] + R[m + 1][i] * q3[1] + R[m + 2][i] * q3[2];
				R[m + 1][i] = y;
				R[m][i] = x;
			}
			for (i = 0; i < k; ++i) {
				x = R[i][m] * q1[0] + R[i][m + 1] * q1[1] + R[i][m + 2] * q1[2];
				y = R[i][m] * q2[0] + R[i][m + 1] * q2[1] + R[i][m + 2] * q2[2];
				R[i][m + 2] = R[i][m] * q3[0] + R[i][m + 1] * q3[1] + R[i][m + 2] * q3[2];
				R[i][m + 1] = y;
				R[i][m] = x;
			}

			for (r = m; r < n - 3; ++r) {
				sum = R[r + 1][r] * R[r + 1][r] + R[r + 2][r] * R[r + 2][r] + R[r + 3][r] * R[r + 3][r];
				l = sqrt(sum);
				if (R[r + 1][r] < 0.0)l = -l;

				//定义w
				x = R[r + 1][r] + l;
				l = 1.0 / sqrt(l * x);
				x = x * l;
				y = R[r + 2][r] * l;
				z = R[r + 3][r] * l;

				//定义Q
				for (i = 0; i < Q.rows; ++i) {
					sum = Q[i][r + 1] * x + Q[i][r + 2] * y + Q[i][r + 3] * z;
					Q[i][r + 1] -= x * sum;
					Q[i][r + 2] -= y * sum;
					Q[i][r + 3] -= z * sum;
				}
				//定义R
				if (r + 5 < n) k = r + 5;
				else k = n;
				for (i = 0; i < k; ++i) {
					sum = R[i][r + 1] * x + R[i][r + 2] * y + R[i][r + 3] * z;
					R[i][r + 1] -= x * sum;
					R[i][r + 2] -= y * sum;
					R[i][r + 3] -= z * sum;
				}
				for (j = r; j < R.cols; ++j){
					sum = R[r + 1][j] * x + R[r + 2][j] * y + R[r + 3][j] * z;
					R[r + 1][j] -= x * sum;
					R[r + 2][j] -= y * sum;
					R[r + 3][j] -= z * sum;
				}
			}

			sum = R[r + 2][r] / R[r + 1][r];
			x = 1 / sqrt(sum * sum + 1);
			y = sum * x;

			for (i = 0; i < n; ++i) {
				sum = R[i][r + 1] * x + R[i][r + 2] * y;
				R[i][r + 2] = R[i][r + 2] * x - R[i][r + 1] * y;
				R[i][r + 1] = sum;
			}
			for (j = r; j < R.cols; ++j) {
				sum = R[r + 1][j] * x + R[r + 2][j] * y;
				R[r + 2][j] = R[r + 2][j] * x - R[r + 1][j] * y;
				R[r + 1][j] = sum;
			}
			for (i = 0; i < Q.rows; ++i) {
				sum = Q[i][r + 1] * x + Q[i][r + 2] * y;
				Q[i][r + 2] = Q[i][r + 2] * x - Q[i][r + 1] * y;
				Q[i][r + 1] = sum;
			}

		}
		else {

			sum = R[m + 1][m] / (R[m][m] - R[n - 1][n - 1]);
			x = 1 / sqrt(sum * sum + 1);
			y = sum * x;

			if (m + 3 < n) k = m + 3;
			else k = n;
			for (i = 0; i < k; ++i) {
				sum = R[i][m] * x + R[i][m + 1] * y;
				R[i][m + 1] = R[i][m + 1] * x - R[i][m] * y;
				R[i][m] = sum;
			}
			for (j = m; j < R.cols; ++j) {
				sum = R[m][j] * x + R[m + 1][j] * y;
				R[m + 1][j] = R[m + 1][j] * x - R[m][j] * y;
				R[m][j] = sum;
			}
			for (i = 0; i < Q.rows; ++i) {
				sum = Q[i][m] * x + Q[i][m + 1] * y;
				Q[i][m + 1] = Q[i][m + 1] * x - Q[i][m] * y;
				Q[i][m] = sum;
			}

			for (r = 0; r < n - 2; ++r) {
				sum = R[r + 2][r] / R[r + 1][r];
				x = 1 / sqrt(sum * sum + 1);
				y = sum * x;

				if (r + 4 < n) k = r + 4;
				else k = n;
				for (i = 0; i < k; ++i) {
					sum = R[i][r + 1] * x + R[i][r + 2] * y;
					R[i][r + 2] = R[i][r + 2] * x - R[i][r + 1] * y;
					R[i][r + 1] = sum;
				}
				for (j = r; j < R.cols; ++j) {
					sum = R[r + 1][j] * x + R[r + 2][j] * y;
					R[r + 2][j] = R[r + 2][j] * x - R[r + 1][j] * y;
					R[r + 1][j] = sum;
				}
				for (i = 0; i < Q.rows; ++i) {
					sum = Q[i][r + 1] * x + Q[i][r + 2] * y;
					Q[i][r + 2] = Q[i][r + 2] * x - Q[i][r + 1] * y;
					Q[i][r + 1] = sum;
				}
			}
		}
	}
}

//*

double _mm256_reduce_add_pd(__m256d vec) {
	double result[3];
	_mm256_store_pd(result, vec);
	return result[0] + result[1] + result[2];
}

void __Schur_QR_R(MAT<double>& Q, MAT<double>& R, int m, int n, double eps) {
	int i, j, k, r;
	double sum, l, x, y;
	complex<double> shift;
	__m256d __temp;
	double __data[12];
	__m256d* data = (__m256d*)__data; //q1, q2, q3 // temp, w

	__data[3] = 0; __data[7] = 0; __data[11] = 0;

	while (1) {

		if (fabs(R[m + 1][m]) < eps) {
			++m;
		}
		for (r = n - 1; r > m; --r) {
			if (fabs(R[r][r - 1]) < eps) {
				if (n - r <= 2) {
					n = r;
					break;
				}
				else {
					__Schur_QR_R(Q, R, r, n, eps);
					n = r;
					break;
				}
			}
		}

		x = R[n - 2][n - 2] + R[n - 1][n - 1];
		y = x * x - (R[n - 2][n - 2] * R[n - 1][n - 1] - R[n - 2][n - 1] * R[n - 1][n - 2]) * 4;
		if (n - m <= 2) {
			if (n - m < 2) {
				break;
			}
			else if (y < 0) {
				break;
			}
		}

		if (y < 0) {
			shift.real = x * 0.5;
			shift.imag = sqrt(-y) * 0.5;

			__data[0] = R[m][m] * R[m][m] + R[m][m + 1] * R[m + 1][m] - 2 * shift.real * R[m][m] + shift.len2();
			__data[1] = R[m + 1][m] * (R[m][m] + R[m + 1][m + 1] - 2 * shift.real);
			__data[2] = R[m + 1][m] * R[m + 2][m + 1];

			l = sqrt(_mm256_reduce_add_pd(_mm256_mul_pd(data[0], data[0])));
			data[0] = _mm256_mul_pd(data[0], _mm256_set1_pd(1.0 / l));

			data[1] = _mm256_sub_pd(_mm256_set_pd(0, 0, 1, 0), _mm256_mul_pd(data[0], _mm256_set1_pd(__data[1])));
			l = sqrt(_mm256_reduce_add_pd(_mm256_mul_pd(data[1], data[1])));
			data[1] = _mm256_mul_pd(data[1], _mm256_set1_pd(1.0 / l));

			__data[8] = __data[1] * __data[6] - __data[2] * __data[5];
			__data[9] = __data[2] * __data[4] - __data[0] * __data[6];
			__data[10] = __data[0] * __data[5] - __data[1] * __data[4];

			for (i = 0; i < Q.rows; ++i) {
				__temp = *(__m256d*)(Q[i] + m);
				Q[i][m] = _mm256_reduce_add_pd(_mm256_mul_pd(__temp, data[0]));
				Q[i][m + 1] = _mm256_reduce_add_pd(_mm256_mul_pd(__temp, data[1]));
				Q[i][m + 2] = _mm256_reduce_add_pd(_mm256_mul_pd(__temp, data[2]));
			}

			if (m + 4 < n) k = m + 4;
			else k = n;
			for (i = m; i < R.cols; ++i) {
				__temp = _mm256_set_pd(0, R[m + 2][i], R[m + 1][i], R[m][i]);
				R[m][i] = _mm256_reduce_add_pd(_mm256_mul_pd(__temp, data[0]));
				R[m + 1][i] = _mm256_reduce_add_pd(_mm256_mul_pd(__temp, data[1]));
				R[m + 2][i] = _mm256_reduce_add_pd(_mm256_mul_pd(__temp, data[2]));
			}
			for (i = 0; i < k; ++i) {
				__temp = *(__m256d*)(R[i] + m);
				R[i][m] = _mm256_reduce_add_pd(_mm256_mul_pd(__temp, data[0]));
				R[i][m + 1] = _mm256_reduce_add_pd(_mm256_mul_pd(__temp, data[1]));
				R[i][m + 2] = _mm256_reduce_add_pd(_mm256_mul_pd(__temp, data[2]));
			}

			for (r = m; r < n - 3; ++r) {
				data[0] = _mm256_set_pd(0, R[r + 3][r], R[r + 2][r], R[r + 1][r]);
				l = sqrt(_mm256_reduce_add_pd(_mm256_mul_pd(data[0], data[0])));
				if (R[r + 1][r] < 0.0)l = -l;
				__data[0] += l;
				data[0] = _mm256_mul_pd(data[0], _mm256_set1_pd(1.0 / sqrt(l * __data[0])));

				for (i = 0; i < Q.rows; ++i) {
					__temp = *(__m256d*)(Q[i] + r + 1);
					*(__m256d*)(Q[i] + r + 1) = _mm256_sub_pd(__temp, _mm256_mul_pd(_mm256_set1_pd(_mm256_reduce_add_pd(_mm256_mul_pd(__temp, data[0]))), data[0]));
				}
				if (r + 5 < n) k = r + 5;
				else k = n;
				for (i = 0; i < k; ++i) {
					__temp = *(__m256d*)(R[i] + r + 1);
					*(__m256d*)(R[i] + r + 1) = _mm256_sub_pd(__temp, _mm256_mul_pd(_mm256_set1_pd(_mm256_reduce_add_pd(_mm256_mul_pd(__temp, data[0]))), data[0]));
				}
				for (j = r; j < R.cols; ++j){
					__temp = _mm256_set_pd(0, R[r + 3][j], R[r + 2][j], R[r + 1][j]);
					data[1] = _mm256_sub_pd(__temp, _mm256_mul_pd(_mm256_set1_pd(_mm256_reduce_add_pd(_mm256_mul_pd(__temp, data[0]))), data[0]));
					R[r + 1][j] = __data[4];
					R[r + 2][j] = __data[5];
					R[r + 3][j] = __data[6];
				}
			}

			sum = R[r + 2][r] / R[r + 1][r];
			x = 1 / sqrt(sum * sum + 1);
			y = sum * x;

			for (i = 0; i < n; ++i) {
				sum = R[i][r + 1] * x + R[i][r + 2] * y;
				R[i][r + 2] = R[i][r + 2] * x - R[i][r + 1] * y;
				R[i][r + 1] = sum;
			}
			for (j = r; j < R.cols; ++j) {
				sum = R[r + 1][j] * x + R[r + 2][j] * y;
				R[r + 2][j] = R[r + 2][j] * x - R[r + 1][j] * y;
				R[r + 1][j] = sum;
			}
			for (i = 0; i < Q.rows; ++i) {
				sum = Q[i][r + 1] * x + Q[i][r + 2] * y;
				Q[i][r + 2] = Q[i][r + 2] * x - Q[i][r + 1] * y;
				Q[i][r + 1] = sum;
			}

		}
		else {
			sum = R[m + 1][m] / (R[m][m] - R[n - 1][n - 1]);
			x = 1 / sqrt(sum * sum + 1);
			y = sum * x;

			if (m + 3 < n) k = m + 3;
			else k = n;
			for (i = 0; i < k; ++i) {
				sum = R[i][m] * x + R[i][m + 1] * y;
				R[i][m + 1] = R[i][m + 1] * x - R[i][m] * y;
				R[i][m] = sum;
			}
			for (j = m; j < R.cols; ++j) {
				sum = R[m][j] * x + R[m + 1][j] * y;
				R[m + 1][j] = R[m + 1][j] * x - R[m][j] * y;
				R[m][j] = sum;
			}
			for (i = 0; i < Q.rows; ++i) {
				sum = Q[i][m] * x + Q[i][m + 1] * y;
				Q[i][m + 1] = Q[i][m + 1] * x - Q[i][m] * y;
				Q[i][m] = sum;
			}

			for (r = 0; r < n - 2; ++r) {
				sum = R[r + 2][r] / R[r + 1][r];
				x = 1 / sqrt(sum * sum + 1);
				y = sum * x;

				if (r + 4 < n) k = r + 4;
				else k = n;
				for (i = 0; i < k; ++i) {
					sum = R[i][r + 1] * x + R[i][r + 2] * y;
					R[i][r + 2] = R[i][r + 2] * x - R[i][r + 1] * y;
					R[i][r + 1] = sum;
				}
				for (j = r; j < R.cols; ++j) {
					sum = R[r + 1][j] * x + R[r + 2][j] * y;
					R[r + 2][j] = R[r + 2][j] * x - R[r + 1][j] * y;
					R[r + 1][j] = sum;
				}
				for (i = 0; i < Q.rows; ++i) {
					sum = Q[i][r + 1] * x + Q[i][r + 2] * y;
					Q[i][r + 2] = Q[i][r + 2] * x - Q[i][r + 1] * y;
					Q[i][r + 1] = sum;
				}
			}
		}
	}
}
//*/