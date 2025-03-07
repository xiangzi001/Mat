
template<typename T>
void __Schur_HouseHolder_R(MAT<T>& X, MAT<T>& R, T* w) {
	int i, j, k, r;
	T sum, cr, hr;
	for (r = 0; r < R.rows - 2; ++r) {
		sum = 0;
		for (k = r + 1; k < R.rows; ++k)
			sum += R[k][r] * R[k][r];
		if (sum == 0.0) {
			continue;
		}
		cr = sqrt(sum);
		if (R[r + 1][r] > 0.0)cr = -cr;
		hr = 1.0 / (cr * (cr - R[r + 1][r]));

		//定义w
		w[r + 1] = R[r + 1][r] - cr;
		for (i = r + 2; i < R.rows; ++i)
		{
			w[i] = R[i][r];
		}

		//定义X
		for (i = 0; i < X.rows; ++i)
		{
			sum = 0;
			for (j = r + 1; j < X.rows; ++j)
				sum += X[i][j] * w[j];
			sum *= hr;
			for (j = r + 1; j < X.cols; ++j)
			{
				X[i][j] -= w[j] * sum;
			}
		}

		//定义R
		for (i = 0; i < R.rows; ++i)
		{
			sum = 0;
			for (j = r + 1; j < R.rows; ++j)
				sum += R[i][j] * w[j];
			sum *= hr;
			for (j = r + 1; j < R.cols; ++j)
			{
				R[i][j] -= w[j] * sum;
			}
		}
		for (j = r; j < R.cols; ++j)
		{
			sum = 0;
			for (i = r + 1; i < R.rows; ++i)
				sum += R[i][j] * w[i];
			sum *= hr;
			for (i = r + 1; i < R.rows; ++i)
				R[i][j] -= w[i] * sum;
		}
	}
}

template<typename T>
bool __Schur_HouseHolder_R2(MAT<T>& X, MAT<T>& R, int m, int n, complex<T> shift, T eps) {

	return false;
}

template<typename T>
void __Schur_Givens_R(MAT<T>& X, MAT<T>& R, int m, int n, T shift) {

	int i, j, k, r;
	T sum, x, y;

}

template<typename T>
void __Schur_QR_R(MAT<T>& Q, MAT<T>& R, MAT<T>& Temp, int m, int n, T eps) {
	int i, j, k, r;
	T sum, l, x, y;
	complex<T> shift;
	T* w = Temp[3];

	while (n - m > 2) {

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
					__Schur_QR_R(Q, R, Temp, r, n, eps);
					n = r;
					break;
				}
			}
		}
		if (n - m <= 2) {
			break;
		}


		x = R[n - 2][n - 2] + R[n - 1][n - 1];
		y = x * x - (R[n - 2][n - 2] * R[n - 1][n - 1] - R[n - 2][n - 1] * R[n - 1][n - 2]) * 4;

		if ( y < 0 && n - m > 3) {
			shift.real = x * 0.5;
			shift.imag = sqrt(-y) * 0.5;

			w[0] = R[m][m] * R[m][m] + R[m][m + 1] * R[m + 1][m] - 2 * shift.real * R[m][m] + shift.len2();
			w[1] = R[m + 1][m] * (R[m][m] + R[m + 1][m + 1] - 2 * shift.real);
			w[2] = R[m + 1][m] * R[m + 2][m + 1];
			l = 1.0 / sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
			Temp[0][0] = w[0] * l;
			Temp[0][1] = w[1] * l;
			Temp[0][2] = w[2] * l;

			l = Temp[0][1];
			for (k = 0; k < 3; ++k) {
				Temp[1][k] = -Temp[0][k] * l;
			}
			Temp[1][1] += 1;

			l = 1.0 / sqrt(Temp[1][0] * Temp[1][0] + Temp[1][1] * Temp[1][1] + Temp[1][2] * Temp[1][2]);
			for (k = 0; k < 3; ++k) {
				Temp[1][k] *= l;
			}

			l = Temp[0][2];
			for (k = 0; k < 3; ++k) {
				Temp[2][k] = -Temp[0][k] * l;
			}
			Temp[2][2] += 1;
			l = 0;
			for (k = 0; k < 3; ++k) {
				l += Temp[1][k] * Temp[2][k];
			}
			for (k = 0; k < 3; ++k) {
				Temp[2][k] -= Temp[1][k] * l;
			}

			l = 1.0 / sqrt(Temp[2][0] * Temp[2][0] + Temp[2][1] * Temp[2][1] + Temp[2][2] * Temp[2][2]);
			for (k = 0; k < 3; ++k) {
				Temp[2][k] *= l;
			}

			for (i = 0; i < Q.rows; ++i) {
				for (j = 0; j < 3; ++j) {
					w[j] = 0;
					for (r = 0; r < 3; ++r) {
						w[j] += Q[i][m + r] * Temp[j][r];
					}
				}
				for (j = 0; j < 3; ++j) {
					Q[i][m + j] = w[j];
				}
			}

			if (m + 4 < n) k = m + 4;
			else k = n;
			for (i = m; i < R.cols; ++i) {
				for (j = 0; j < 3; ++j) {
					w[j] = 0;
					for (r = 0; r < 3; ++r) {
						w[j] += R[m + r][i] * Temp[j][r];
					}
				}
				for (j = 0; j < 3; ++j) {
					R[m + j][i] = w[j];
				}
			}
			for (i = 0; i < k; ++i) {
				for (j = 0; j < 3; ++j) {
					w[j] = 0;
					for (r = 0; r < 3; ++r) {
						w[j] += R[i][m + r] * Temp[j][r];
					}
				}
				for (j = 0; j < 3; ++j) {
					R[i][m + j] = w[j];
				}
			}

			for (r = m; r < n - 3; ++r) {
				sum = R[r + 1][r] * R[r + 1][r]
					+ R[r + 2][r] * R[r + 2][r]
					+ R[r + 3][r] * R[r + 3][r];
				l = sqrt(sum);
				if (R[r + 1][r] < 0.0)l = -l;

				//定义w
				w[0] = R[r + 1][r] + l;
				w[1] = R[r + 2][r];
				w[2] = R[r + 3][r];

				l = 1.0 / (l * (l + R[r + 1][r]));

				//cout << l << endl;
				//定义Q
				for (i = 0; i < Q.rows; ++i) {
					sum = (Q[i][r + 1] * w[0]
						+ Q[i][r + 2] * w[1]
						+ Q[i][r + 3] * w[2]) * l;
					Q[i][r + 1] -= w[0] * sum;
					Q[i][r + 2] -= w[1] * sum;
					Q[i][r + 3] -= w[2] * sum;
				}
				//定义R
				if (r + 5 < n) k = r + 5;
				else k = n;
				for (i = 0; i < k; ++i) {
					sum = (R[i][r + 1] * w[0]
						+ R[i][r + 2] * w[1]
						+ R[i][r + 3] * w[2]) * l;
					R[i][r + 1] -= w[0] * sum;
					R[i][r + 2] -= w[1] * sum;
					R[i][r + 3] -= w[2] * sum;
				}
				for (j = r; j < R.cols; ++j)
				{
					sum = (R[r + 1][j] * w[0]
						+ R[r + 2][j] * w[1]
						+ R[r + 3][j] * w[2]) * l;
					R[r + 1][j] -= w[0] * sum;
					R[r + 2][j] -= w[1] * sum;
					R[r + 3][j] -= w[2] * sum;
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
		//print(R);
	}
}
