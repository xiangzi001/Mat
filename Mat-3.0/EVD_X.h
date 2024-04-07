

template<typename T>
void EVD_X_HouseHolder(MAT<complex<T>>& X, MAT<complex<T>>& R, complex<T>*& w) {

	int i, j, k, r;

	complex<T> sum, _sign;
	T temp, cr, hr;

	for (r = 0; r < R.rows - 2; ++r) {

		temp = 0;
		for (k = r+1; k < R.rows; ++k) {
			temp += R[k][r].len2();
		}
		if (temp == 0) {
			continue;
		}
		cr = sqrt(temp);
		_sign = sign(R[r+1][r]) * cr;

		//if (R[r][r].real < 0.0)cr = -cr;
		hr = 1 / (cr * cr - times_conj(_sign, R[r+1][r]).real);

		//定义w
		w[r+1] = R[r+1][r] - _sign;
		for (i = r + 2; i < R.rows; ++i)
		{
			w[i] = R[i][r];
		}

		for (i = 0; i < X.rows; ++i)//定义X
		{
			sum = 0;
			for (j = r+1; j < X.rows; ++j)
				sum += X[i][j] * w[j];
			sum = sum * hr;
			for (j = r+1; j < X.cols; ++j)
			{
				X[i][j] -= times_conj(sum, w[j]);
			}
		}
		for (j = 0; j < R.cols; ++j)//定义R
		{
			sum = 0;
			for (i = r+1; i < R.rows; ++i)
				sum += times_conj(R[i][j], w[i]);
			sum = sum * hr;
			for (i = r+1; i < R.rows; ++i)
				R[i][j] -= w[i] * sum;
		}
		for (i = 0; i < R.rows; ++i)//定义R
		{
			sum = 0;
			for (j = r + 1; j < R.rows; ++j)
				sum += R[i][j] * w[j];
			sum = sum * hr;
			for (j = r + 1; j < R.cols; ++j)
			{
				R[i][j] -= times_conj(sum, w[j]);
			}
		}
	}
}

template<typename T>
void EVD_X_Givens(MAT<complex<T>>& X, MAT<complex<T>>& R, int m, int n, double eps, complex<T>*& Temp) {

	T c;
	complex<T> temp, t, s, _s;
	int i, k;
	while (n - m - 1) {

		if (fabs(R[m + 1][m]) < eps) {
			m++;
			continue;
		}

		t = R[m + 1][m] / R[m][m];
		c = 1 / sqrt(t.real * t.real + t.imag * t.imag + 1);
		s = t * c;
		_s = s.conj();

		for (k = m; k < R.rows; ++k) {
			temp = R[m][k] * c + R[m + 1][k] * _s;
			R[m + 1][k] = R[m + 1][k] * c - R[m][k] * s;
			R[m][k] = temp;
		}
		for (k = 0; k < R.rows; ++k) {
			temp = R[k][m] * c + R[k][m + 1] * s;
			R[k][m + 1] = R[k][m + 1] * c - R[k][m] * _s;
			R[k][m] = temp;
		}
		for (k = 0; k < X.rows; ++k) {
			temp = X[k][m] * c + X[k][m + 1] * s;
			X[k][m + 1] = X[k][m + 1] * c - X[k][m] * _s;
			X[k][m] = temp;
		}

		for (i = m + 2; i < n; ++i) {
			if (fabs(R[i][i - 1]) < eps) {
				if (n - i - 1) {
					EVD_X_Givens(X, R, m, i, eps, Temp);
					m = i;
					break;
				}
				else {
					n--;
				}
			}
			else {
				t = R[i][i - 2] / R[i - 1][i - 2];
				c = 1 / sqrt(t.real * t.real + t.imag * t.imag + 1);
				s = t * c;
				_s = s.conj();

				for (k = m; k < R.rows; ++k) {
					temp = R[i - 1][k] * c + R[i][k] * _s;
					R[i][k] = R[i][k] * c - R[i - 1][k] * s;
					R[i - 1][k] = temp;
				}
				for (k = 0; k < R.rows; ++k) {
					temp = R[k][i - 1] * c + R[k][i] * s;
					R[k][i] = R[k][i] * c - R[k][i - 1] * _s;
					R[k][i - 1] = temp;
				}
				for (k = 0; k < X.rows; ++k) {
					temp = X[k][i - 1] * c + X[k][i] * s;
					X[k][i] = X[k][i] * c - X[k][i - 1] * _s;
					X[k][i - 1] = temp;
				}
			}
		}
		for (int i = m; i < n; ++i) {
			Temp[i] = Temp[i] + R[n - 1][n - 1];
			R[i][i] = R[i][i] - R[n - 1][n - 1];
		}
	}
}

template<typename T>
void EVD_X_QR(MAT<complex<T>>& A, MAT<complex<T>>& X, double eps) {
	int i = 0;
	for (i = 0; i < X.rows * X.cols; ++i) {
		X.data[i] = 0;
	}
	for (i = 0; i < X.rows; ++i) {
		X[i][i] = 1;
	}

	complex<T>* w = new complex<T>[A.rows];
	EVD_X_HouseHolder(X, A, w);

	for (int i = 0; i < A.rows; ++i) {
		w[i] = 0;
	}
	EVD_X_Givens(X, A, 0, A.rows, eps, w);

	for (int i = 0; i < A.rows; ++i) {
		A[i][i] = A[i][i] + w[i];
	}

	delete[] w;
}

template<typename T>
void EVD_X(MAT<complex<T>>& A, MAT<complex<T>>& E, MAT<complex<T>>& X, double eps) {
	for (int i = 0; i < E.rows * E.cols; ++i) {
		E.data[i] = A.data[i];
	}
	EVD_X_QR(E, X, eps);
}
