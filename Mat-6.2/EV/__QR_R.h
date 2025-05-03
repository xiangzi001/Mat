
template<typename T>
bool __EV_R_BalanceMatrix(MAT<T>& R, T* w, T* v, double eps) {
	int i = 0, j;
	double temp;
	bool ret = false;

	for (i = 0; i < R.rows; ++i) {
		v[i] = -R[i][i] * R[i][i];
	}
	for (i = 0; i < R.rows; ++i) {
		temp = 0;
		for (j = 0; j < R.rows; ++j) {
			temp += R[i][j] * R[i][j];
		}
		w[i] = temp + v[i];
	}
	for (i = 0; i < R.rows; ++i) {
		for (j = 0; j < R.rows; ++j) {
			v[j] += R[i][j] * R[i][j];
		}
	}

	for (i = 0; i < R.rows; ++i) {
		w[i] = pow(w[i] / v[i], 0.25);
		ret = ret || (fabs(w[i] - 1) > eps);
	}

	for (i = 0; i < R.rows; ++i) {
		for (j = 0; j < R.rows; ++j) {
			R[i][j] *= w[j];
		}
	}
	for (i = 0; i < R.rows; ++i) {
		temp = 1.0 / w[i];
		for (j = 0; j < R.rows; ++j) {
			R[i][j] *= temp;
		}
	}
	return ret;
}

bool __EV_R_BalanceMatrix(MAT<double>& R, double* w, double* v, double eps) {
	int i = 0, j;
	int n = R.cols / 8;
	int m = R.cols - n * 8;
	__m512d add;
	__m512d* temp1;
	__m512d* temp2;

	__mmask8 mask = (1 << m) - 1;

	for (i = 0; i < R.rows; ++i) {
		v[i] = -R[i][i] * R[i][i];
	}

	for (i = 0; i < R.rows; ++i) {
		add = _mm512_mask_blend_pd(mask, _mm512_setzero_pd(), _mm512_mul_pd(*(__m512d*)(R[i]), *(__m512d*)(R[i])));
		temp1 = (__m512d*)(R[i] + m);
		for (j = 0; j < n; ++j) {
			add = _mm512_add_pd(add, _mm512_mul_pd(temp1[j], temp1[j]));
		}
		w[i] = _mm512_reduce_add_pd(add) + v[i];
	}
	temp2 = (__m512d*)(v + m);
	for (i = 0; i < R.rows; ++i) {
		*(__m512d*)(v) = _mm512_add_pd(*(__m512d*)(v), _mm512_mask_blend_pd(mask, _mm512_setzero_pd(), _mm512_mul_pd(*(__m512d*)(R[i]), *(__m512d*)(R[i]))));
		temp1 = (__m512d*)(R[i] + m);
		for (j = 0; j < n; ++j) {
			temp2[j] = _mm512_add_pd(temp2[j], _mm512_mul_pd(temp1[j], temp1[j]));
		}
	}

	add = _mm512_set1_pd(0.25);
	temp1 = (__m512d*)w;
	temp2 = (__m512d*)v;
	n = (R.cols - 1) / 8 + 1;
	for (i = 0; i < n; ++i) {
		temp1[i] = _mm512_pow_pd(_mm512_div_pd(temp1[i], temp2[i]), add);
	}
	for (i = R.rows; i < 8 * n; ++i) {
		w[i] = 1;
	}

	for (i = 0; i < R.rows; ++i) {
		temp2 = (__m512d*)(R[i]);
		for (j = 0; j < n; ++j) {
			temp2[j] = _mm512_mul_pd(temp1[j], temp2[j]);
		}
	}
	n = R.cols / 8;
	for (i = 0; i < R.rows; ++i) {
		add = _mm512_set1_pd(1.0 / w[i]);
		*(__m512d*)(R[i]) = _mm512_mul_pd(*(__m512d*)(R[i]), _mm512_mask_blend_pd(mask, _mm512_set1_pd(1), add));
		temp2 = (__m512d*)(R[i] + m);
		for (j = 0; j < n; ++j) {
			temp2[j] = _mm512_mul_pd(add, temp2[j]);
		}
	}
	for (i = 0; i < R.rows; ++i) {
		if (w[i] - 1 > eps)
			return true;
	}
	return false;
}


template<typename T>
void __EV_HouseHolder_R(MAT<T>& R, T* w, T* v) {
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

		for (i = 0; i < R.rows; ++i)//¶¨ÒåR
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


void __EV_HouseHolder_R(MAT<double>& R, double* w, double* v) {

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

void __EV_HouseHolder_R(MAT<float>& R, float* w, float* v) {

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


template<typename T>
void __EV_QR_R(MAT<T>& R, int m, int n, double eps) {
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
					__EV_QR_R(R, r, n, eps);
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

				x = R[r + 1][r] + l;
				l = 1.0 / sqrt(l * x);
				x = x * l;
				y = R[r + 2][r] * l;
				z = R[r + 3][r] * l;

				if (r + 5 < n) k = r + 5;
				else k = n;
				for (i = 0; i < k; ++i) {
					sum = R[i][r + 1] * x + R[i][r + 2] * y + R[i][r + 3] * z;
					R[i][r + 1] -= x * sum;
					R[i][r + 2] -= y * sum;
					R[i][r + 3] -= z * sum;
				}
				for (j = r; j < R.cols; ++j) {
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
			}
		}
	}
}

void __EV_QR_R(MAT<double>& R, int m, int n, double eps) {
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
					__EV_QR_R(R, r, n, eps);
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

				if (r + 5 < n) k = r + 5;
				else k = n;
				for (i = 0; i < k; ++i) {
					__temp = *(__m256d*)(R[i] + r + 1);
					*(__m256d*)(R[i] + r + 1) = _mm256_sub_pd(__temp, _mm256_mul_pd(_mm256_set1_pd(_mm256_reduce_add_pd(_mm256_mul_pd(__temp, data[0]))), data[0]));
				}
				for (j = r; j < R.cols; ++j) {
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
			}
		}
	}
}
//*/