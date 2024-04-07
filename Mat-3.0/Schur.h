
template<typename t>
void Schur_HouseHolder(MAT<complex<t>>& U, MAT<complex<t>>& R, complex<t>*& w) {

	int i, j, k, r;

	complex<t> sum, _sign;
	t temp, cr, hr;

	for (r = 0; r < R.rows - 2; ++r) {

		temp = 0;
		for (k = r + 1; k < R.rows; ++k) {
			temp += R[k][r].len2();
		}
		if (temp == 0) {
			continue;
		}
		cr = sqrt(temp);
		_sign = sign(R[r + 1][r]) * cr;

		//if (R[r][r].real < 0.0)cr = -cr;
		hr = 1 / (cr * cr - times_conj(_sign, R[r + 1][r]).real);

		//定义w
		w[r + 1] = R[r + 1][r] - _sign;
		for (i = r + 2; i < R.rows; ++i)
		{
			w[i] = R[i][r];
		}

		for (i = 0; i < U.rows; ++i)//定义U
		{
			sum = 0;
			for (j = r + 1; j < U.rows; ++j)
				sum += U[i][j] * w[j];
			sum = sum * hr;
			for (j = r + 1; j < U.cols; ++j)
			{
				U[i][j] -= times_conj(sum, w[j]);
			}
		}
		for (j = 0; j < R.cols; ++j)//定义R
		{
			sum = 0;
			for (i = r + 1; i < R.rows; ++i)
				sum += times_conj(R[i][j], w[i]);
			sum = sum * hr;
			for (i = r + 1; i < R.rows; ++i)
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

template<typename t>
void Schur_Givens(MAT<complex<t>>& U, MAT<complex<t>>& R, int m, int n, double eps, complex<t>*& Temp) {

	t c;
	complex<t> temp, t, s, _s;
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
		for (k = 0; k < U.rows; ++k) {
			temp = U[k][m] * c + U[k][m + 1] * s;
			U[k][m + 1] = U[k][m + 1] * c - U[k][m] * _s;
			U[k][m] = temp;
		}

		for (i = m + 2; i < n; ++i) {
			if (fabs(R[i][i - 1]) < eps) {
				if (n - i - 1) {
					Schur_Givens(U, R, m, i, eps, Temp);
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
				for (k = 0; k < U.rows; ++k) {
					temp = U[k][i - 1] * c + U[k][i] * s;
					U[k][i] = U[k][i] * c - U[k][i - 1] * _s;
					U[k][i - 1] = temp;
				}
			}
		}
		/*
		t = R[n - 2][n - 2] + R[n - 1][n - 1];
		s = R[n - 2][n - 2] * R[n - 1][n - 1] - R[n - 2][n - 1] * R[n - 1][n - 2];
		s = sqrt(t * t - s * 4);
		temp = t - R[n - 1][n - 1] * 2;
		if ((temp+s).len2() > (temp-s).len2()) {
			temp = (t - s) * 0.5;
		}
		else {
			temp = (t + s) * 0.5;
		}
		//*/
		temp = R[n - 1][n - 1];
		for (int i = m; i < n; ++i) {
			Temp[i] += temp;
			R[i][i] -= temp;
		}
	}
}

template<typename t>
void Schur(MAT<complex<t>>& A, MAT<complex<t>>& T, MAT<complex<t>>& U, double eps) {
	int i;
	int s(T.rows * T.cols);
	for (int i = 0; i < s; ++i) {
		T.data[i] = A.data[i];
	}
	s = U.rows * U.cols;
	for (i = 0; i < s; ++i) {
		U.data[i] = 0;
	}
	for (i = 0; i < U.rows; ++i) {
		U[i][i] = 1;
	}

	complex<t>* w = new complex<t>[A.rows];

	Schur_HouseHolder(U, T, w);

	for (int i = 0; i < T.rows; ++i) {
		w[i] = 0;
	}

	Schur_Givens(U, T, 0, T.rows, eps, w);

	for (int i = 0; i < T.rows; ++i) {
		T[i][i] = T[i][i] + w[i];
	}

	delete[] w;
}
