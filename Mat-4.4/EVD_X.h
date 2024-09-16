
template<typename T>
void EVD_UP(MAT<T>& A, MAT<T>& X, T* w) {
	int i, j, k;
	double len;
	T sum;
	T* temp;
	for (i = A.cols - 1; i > 0; --i) {
		for (j = i - 1; j >= 0; --j) {
			temp = A[j];
			sum = 0;
			for (k = j + 1; k < i; ++k) {
				sum += temp[k] * w[k];
			}
			sum += temp[i];
			w[j] = sum / (A[i][i] - temp[j]);
		}

		len = 0;
		for (j = 0; j < i; ++j)
			len += len2(w[j]);
		len = 1 / sqrt(len + 1);
		for (j = 0; j < i; ++j)
			w[j] *= len;
		w[i] = len;

		for (j = 0; j < X.rows; ++j) {
			temp = X[j];
			sum = 0;
			for (k = 0; k <= i; ++k) {
				sum += temp[k] * w[k];
			}
			temp[i] = sum;
		}
	}
	for (i = 0; i < A.rows; ++i) {
		temp = A[i];
		for (j = i + 1; j < A.cols; ++j) {
			temp[j] = 0;
		}
	}
}

template<typename T>
void EVD_X_HouseHolder(MAT<complex<T>>& X, MAT<complex<T>>& R, T* w) {
	int i, j, k, r;
	T sum, cr, hr;
	for (r = 0; r < R.rows - 2; ++r) {
		sum = 0;
		for (k = r + 1; k < R.rows; ++k)
			sum += R[k][r].real * R[k][r].real;
		if (sum == 0.0) {
			continue;
		}
		cr = sqrt(sum);
		if (R[r + 1][r].real > 0.0)cr = -cr;
		hr = 1.0 / ( cr * (cr - R[r + 1][r].real) );

		//定义w
		w[r + 1] = R[r + 1][r].real - cr;
		for (i = r + 2; i < R.rows; ++i)
		{
			w[i] = R[i][r].real;
		}

		//定义X
		for (i = 0; i < X.rows; ++i)
		{
			sum = 0;
			for (j = r + 1; j < X.rows; ++j)
				sum += X[i][j].real * w[j];
			sum *= hr;
			for (j = r + 1; j < X.cols; ++j)
			{
				X[i][j].real -=  w[j] * sum;
			}
		}

		//定义R
		for (i = 0; i < R.rows; ++i)
		{
			sum = 0;
			for (j = r + 1; j < R.rows; ++j)
				sum += R[i][j].real * w[j];
			sum *= hr;
			for (j = r + 1; j < R.cols; ++j)
			{
				R[i][j].real -= w[j] * sum;
			}
		}
		for (j = r; j < R.cols; ++j)
		{
			sum = 0;
			for (i = r + 1; i < R.rows; ++i)
				sum += R[i][j].real * w[i];
			sum *= hr;
			for (i = r + 1; i < R.rows; ++i)
				R[i][j].real -= w[i] * sum;
		}
	}
}


template<typename T>
void EVD_X_QR(MAT<complex<T>>& X, MAT<complex<T>>& R, int m, int n, T eps) {
	int i, j, k, r;
	T sum, l, x, y, z;
	complex<T> shift;

	while (n - m > 2) {

		if (fabs(R[m + 1][m].real) < eps) {
			++m;
		}

		x = R[n - 2][n - 2].real + R[n - 1][n - 1].real;
		y = x * x - (R[n - 2][n - 2].real * R[n - 1][n - 1].real - R[n - 2][n - 1].real * R[n - 1][n - 2].real) * 4;
		if (y < 0 && m - n > 3) {
			shift.real = x * 0.5;
			shift.imag = sqrt(-y) * 0.5;


			x = R[m][m].real * R[m][m].real + R[m][m + 1].real * R[m + 1][m].real - 2 * shift.real * R[m][m].real + shift.len2();
			y = R[m + 1][m].real * (R[m][m].real + R[m + 1][m + 1].real - 2 * shift.real);
			z = R[m + 1][m].real * R[m + 2][m + 1].real;
			l = 1.0 / sqrt(x * x + y * y + z * z);
			x = sqrt((1 - x * l) * 0.5);
			y = -(y * l / x) * 0.5;
			z = -(z * l / x) * 0.5;

			for (i = 0; i < X.rows; ++i)
			{
				sum = (X[i][m].real * x
					+ X[i][m + 1].real * y
					+ X[i][m + 2].real * z) * 2;
				X[i][m].real -= x * sum;
				X[i][m + 1].real -= y * sum;
				X[i][m + 2].real -= z * sum;
			}
			if (m + 3 < n) k = m + 3;
			else k = n;
			for (i = 0; i < k; ++i)
			{
				sum = (R[i][m].real * x
					+ R[i][m + 1].real * y
					+ R[i][m + 2].real * z) * 2;
				R[i][m].real -= x * sum;
				R[i][m + 1].real -= y * sum;
				R[i][m + 2].real -= z * sum;
			}
			for (j = m; j < R.cols; ++j)
			{
				sum = (R[m][j].real * x
					+ R[m + 1][j].real * y
					+ R[m + 2][j].real * z) * 2;
				R[m][j].real -= x * sum;
				R[m + 1][j].real -= y * sum;
				R[m + 2][j].real -= z * sum;
			}

			for (r = m; r < n - 3; ++r) {
				sum = R[r + 1][r].real * R[r + 1][r].real
					+ R[r + 2][r].real * R[r + 2][r].real
					+ R[r + 3][r].real * R[r + 3][r].real;
				l = sqrt(sum);
				if (R[r + 1][r].real > 0.0)l = -l;

				//定义w
				x = R[r + 1][r].real - l;
				y = R[r + 2][r].real;
				z = R[r + 3][r].real;

				l = 1.0 / (l * (l - R[r + 1][r].real));
				//定义X
				for (i = 0; i < X.rows; ++i) {
					sum = (X[i][r + 1].real * x
						+ X[i][r + 2].real * y
						+ X[i][r + 3].real * z) * l;
					X[i][r + 1].real -= x * sum;
					X[i][r + 2].real -= y * sum;
					X[i][r + 3].real -= z * sum;
				}
				//定义R
				if (r + 3 < n) k = r + 3;
				else k = n;
				for (i = 0; i < k; ++i) {
					sum = (R[i][r + 1].real * x
						+ R[i][r + 2].real * y
						+ R[i][r + 3].real * z) * l;
					R[i][r + 1].real -= x * sum;
					R[i][r + 2].real -= y * sum;
					R[i][r + 3].real -= z * sum;
				}
				for (j = r; j < R.cols; ++j)
				{
					sum = (R[r + 1][j].real * x
						+ R[r + 2][j].real * y
						+ R[r + 3][j].real * z) * l;
					R[r + 1][j].real -= x * sum;
					R[r + 2][j].real -= y * sum;
					R[r + 3][j].real -= z * sum;
				}
			}

			sum = R[r + 2][r].real / R[r + 1][r].real;
			x = 1 / sqrt(sum * sum + 1);
			y = sum * x;

			for (i = 0; i < n; ++i) {
				sum = R[i][r + 1].real * x + R[i][r + 2].real * y;
				R[i][r + 2].real = R[i][r + 2].real * x - R[i][r + 1].real * y;
				R[i][r + 1].real = sum;
			}
			for (j = r; j < R.cols; ++j) {
				sum = R[r + 1][j].real * x + R[r + 2][j].real * y;
				R[r + 2][j].real = R[r + 2][j].real * x - R[r + 1][j].real * y;
				R[r + 1][j].real = sum;
			}
			for (i = 0; i < X.rows; ++i) {
				sum = X[i][r + 1].real * x + X[i][r + 2].real * y;
				X[i][r + 2].real = X[i][r + 2].real * x - X[i][r + 1].real * y;
				X[i][r + 1].real = sum;
			}
			//cout << "Householder" << endl;
			//cout << m << ",	" << n << endl;
		}
		else {
			//cout << "Givens" << endl;
			//cout << m << ",	" << n << endl;
			sum = R[m + 1][m].real / (R[m][m].real - R[n-1][n-1].real);
			x = 1 / sqrt(sum * sum + 1);
			y = sum * x;

			if (m + 3 < n) k = m + 3;
			else k = n;
			for (i = 0; i < k; ++i) {
				sum = R[i][m].real * x + R[i][m + 1].real * y;
				R[i][m + 1].real = R[i][m + 1].real * x - R[i][m].real * y;
				R[i][m].real = sum;
			}
			for (j = m; j < R.cols; ++j) {
				sum = R[m][j].real * x + R[m + 1][j].real * y;
				R[m + 1][j].real = R[m + 1][j].real * x - R[m][j].real * y;
				R[m][j].real = sum;
			}
			for (i = 0; i < X.rows; ++i) {
				sum = X[i][m].real * x + X[i][m + 1].real * y;
				X[i][m + 1].real = X[i][m + 1].real * x - X[i][m].real * y;
				X[i][m].real = sum;
			}

			for (r = 0; r < n - 2; ++r) {
				sum = R[r + 2][r].real / R[r + 1][r].real;
				x = 1 / sqrt(sum * sum + 1);
				y = sum * x;

				if (r + 4 < n) k = r + 4;
				else k = n;
				for (i = 0; i < k; ++i) {
					sum = R[i][r + 1].real * x + R[i][r + 2].real * y;
					R[i][r + 2].real = R[i][r + 2].real * x - R[i][r + 1].real * y;
					R[i][r + 1].real = sum;
				}
				for (j = r; j < R.cols; ++j) {
					sum = R[r + 1][j].real * x + R[r + 2][j].real * y;
					R[r + 2][j].real = R[r + 2][j].real * x - R[r + 1][j].real * y;
					R[r + 1][j].real = sum;
				}
				for (i = 0; i < X.rows; ++i) {
					sum = X[i][r + 1].real * x + X[i][r + 2].real * y;
					X[i][r + 2].real = X[i][r + 2].real * x - X[i][r + 1].real * y;
					X[i][r + 1].real = sum;
				}
			}
		}
		//print(R);

		for (r = n - 1; r > 0; --r) {
			if (fabs(R[r][r - 1].real) < eps) {
				if (n - r <= 2) {
					n = r;
					break;
				}
				else {
					EVD_X_QR(X, R, r, n, eps);
					n = r;
					break;
				}
			}
		}
	}
}


template<typename t>
void EVD_X_T(MAT<complex<t>>& X, MAT<complex<t>>& R, double eps) {

	t c;
	complex<t> t, s, _s, temp;
	int r, k;
	for (r = 1; r < R.rows; ++r) {

		if (fabs(R[r][r - 1]) < eps) {
			continue;
		}

		t = R[r - 1][r - 1] + R[r][r];
		s = R[r - 1][r - 1] * R[r][r] - R[r - 1][r] * R[r][r - 1];
		s = sqrt(t * t - s * 4);
		temp = t - R[r][r] * 2;
		if ((temp + s).len2() > (temp - s).len2()) {
			temp = (t - s) * 0.5;
		}
		else {
			temp = (t + s) * 0.5;
		}

		t = R[r][r - 1] / (R[r - 1][r - 1] - temp);
		c = 1 / sqrt(t.real * t.real + t.imag * t.imag + 1);
		s = t * c;
		_s = s.conj();

		for (k = r - 1; k < R.rows; ++k) {
			temp = R[r - 1][k] * c + R[r][k] * _s;
			R[r][k] = R[r][k] * c - R[r - 1][k] * s;
			R[r - 1][k] = temp;
		}
		for (k = 0; k < r - 1 + 3; ++k) {
			temp = R[k][r - 1] * c + R[k][r] * s;
			R[k][r] = R[k][r] * c - R[k][r - 1] * _s;
			R[k][r - 1] = temp;
		}
		for (k = 0; k < X.rows; ++k) {
			temp = X[k][r - 1] * c + X[k][r] * s;
			X[k][r] = X[k][r] * c - X[k][r - 1] * _s;
			X[k][r - 1] = temp;
		}

		//*/
		//temp = R[n - 1][n - 1];
	}
}

template<typename t>
void EVD_X(MAT<t>& A, MAT<complex<t>>& E, MAT<complex<t>>& X, double eps) {
	int i;
	int s(E.rows * E.cols);
	for (int i = 0; i < s; ++i) {
		E.data[i] = A.data[i];
	}
	s = X.rows * X.cols;
	for (i = 0; i < s; ++i) {
		X.data[i] = 0;
	}
	for (i = 0; i < X.rows; ++i) {
		X[i][i] = 1;
	}

	complex<t>* w = new complex<t>[A.rows];
	//t* w = new t[A.rows];

	EVD_X_HouseHolder(X, E, (t*)w);
	EVD_X_QR(X, E, 0, E.rows, eps);
	EVD_X_T(X, E, eps);
	EVD_UP(E, X, w);
	//Schur_HouseHolder(X, E, w);

	delete[] w;
}
template<typename t>
void EVD_X(MAT<complex<t>>& A, MAT<complex<t>>& E, MAT<complex<t>>& X, double eps) {
	int i;
	int s(E.rows * E.cols);
	for (int i = 0; i < s; ++i) {
		E.data[i] = A.data[i];
	}
	s = X.rows * X.cols;
	for (i = 0; i < s; ++i) {
		X.data[i] = 0;
	}
	for (i = 0; i < X.rows; ++i) {
		X[i][i] = 1;
	}

	complex<t>* w = new complex<t>[A.rows];

	Schur_HouseHolder(X, E, w);

	Schur_Givens(X, E, 0, E.rows, eps);

	EVD_UP(E, X, w);

	delete[] w;
}
