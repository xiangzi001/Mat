

template<typename t>
void __EV_HouseHolder(MAT<complex<t>>& R, complex<t>*& w) {

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
		sum = R[r + 1][r];
		_sign = sign(sum) * cr;

		//if (R[r][r].real < 0.0)cr = -cr;
		hr = 1 / (cr * cr - _sign.real * sum.real - _sign.imag * sum.imag);

		//定义w
		w[r + 1] = sum - _sign;
		for (i = r + 2; i < R.rows; ++i)
		{
			w[i] = R[i][r];
		}

		for (j = 0; j < R.cols; ++j)//定义R
		{
			sum = 0;
			for (i = r + 1; i < R.rows; ++i)
				sum += times_conj(R[i][j], w[i]);
			sum *= hr;
			for (i = r + 1; i < R.rows; ++i)
				R[i][j] -= w[i] * sum;
		}
		for (i = 0; i < R.rows; ++i)//定义R
		{
			sum = 0;
			for (j = r + 1; j < R.rows; ++j)
				sum += R[i][j] * w[j];
			sum *= hr;
			for (j = r + 1; j < R.cols; ++j)
			{
				R[i][j] -= times_conj(sum, w[j]);
			}
		}
	}
}

template<typename t>
void __EV_Givens(MAT<complex<t>>& R, int m, int n, double eps) {

	t c;
	complex<t> t, s, _s, temp;
	int i, k;
	while (n - m > 1) {

		if (fabs(R[m + 1][m]) < eps) {
			m++;
			continue;
		}

		t = R[m + 1][m] / (R[m][m] - temp);
		c = 1 / sqrt(t.real * t.real + t.imag * t.imag + 1);
		s = t * c;
		_s = s.conj();

		for (k = m; k < R.rows; ++k) {
			temp = R[m][k] * c + R[m + 1][k] * _s;
			R[m + 1][k] = R[m + 1][k] * c - R[m][k] * s;
			R[m][k] = temp;
		}
		for (k = 0; k < m + 3; ++k) {
			temp = R[k][m] * c + R[k][m + 1] * s;
			R[k][m + 1] = R[k][m + 1] * c - R[k][m] * _s;
			R[k][m] = temp;
		}

		for (i = m + 2; i < n; ++i) {
			if (fabs(R[i][i - 1]) < eps) {
				if (n - i == 1) {
					n = n - 1;
					break;
				}
				else {
					__EV_Givens(R, m, i, eps);
					m = i;
				}
			}
			else {
				t = R[i][i - 2] / R[i - 1][i - 2];
				c = 1 / sqrt(t.real * t.real + t.imag * t.imag + 1);
				s = t * c;
				_s = s.conj();

				for (k = i - 2; k < R.rows; ++k) {
					temp = R[i - 1][k] * c + R[i][k] * _s;
					R[i][k] = R[i][k] * c - R[i - 1][k] * s;
					R[i - 1][k] = temp;
				}
				if (i + 2 < R.rows)
					for (k = 0; k < i + 2; ++k) {
						temp = R[k][i - 1] * c + R[k][i] * s;
						R[k][i] = R[k][i] * c - R[k][i - 1] * _s;
						R[k][i - 1] = temp;
					}
				else
					for (k = 0; k < R.rows; ++k) {
						temp = R[k][i - 1] * c + R[k][i] * s;
						R[k][i] = R[k][i] * c - R[k][i - 1] * _s;
						R[k][i - 1] = temp;
					}
			}
		}
		///*
		t = R[n - 2][n - 2] + R[n - 1][n - 1];
		s = R[n - 2][n - 2] * R[n - 1][n - 1] - R[n - 2][n - 1] * R[n - 1][n - 2];
		s = sqrt(t * t - s * 4);
		temp = t - R[n - 1][n - 1] * 2;
		if ((temp + s).len2() > (temp - s).len2()) {
			temp = (t - s) * 0.5;
		}
		else {
			temp = (t + s) * 0.5;
		}
		//*/
		//temp = R[n - 1][n - 1];
	}
}

template<typename T>
void __EV_HouseHolder_R(MAT<T>& R, T* w) {
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
void __EV_R(MAT<T>& R, MAT<T>& Temp, int m, int n, T eps) {
	int i, j, k, r;
	T sum, l, x, y, z;
	complex<T> shift;
	T* q1 = Temp[0];
	T* q2 = Temp[1];
	T* q3 = Temp[2];

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
					__EV_R(R, Temp, r, n, eps);
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

		if (y < 0 && n - m > 3) {
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

			q3[0] = -q1[0] * q1[2];		q3[1] = -q1[1] * q1[2];		q3[2] = 1 - q1[2] * q1[2];
			l = q2[0] * q3[0] + q2[1] * q3[1] + q2[2] * q3[2];
			q3[0] -= q2[0] * l;			q3[1] -= q2[1] * l;			q3[2] -= q2[2] * l;
			l = 1.0 / sqrt(q3[0] * q3[0] + q3[1] * q3[1] + q3[2] * q3[2]);
			q3[0] *= l; q3[1] *= l; q3[2] *= l;

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
				y = R[r + 2][r];
				z = R[r + 3][r];
				l = 1.0 / (l * (l + R[r + 1][r]));

				//定义R
				if (r + 5 < n) k = r + 5;
				else k = n;
				for (i = 0; i < k; ++i) {
					sum = (R[i][r + 1] * x + R[i][r + 2] * y + R[i][r + 3] * z) * l;
					R[i][r + 1] -= x * sum;
					R[i][r + 2] -= y * sum;
					R[i][r + 3] -= z * sum;
				}
				for (j = r; j < R.cols; ++j)
				{
					sum = (R[r + 1][j] * x + R[r + 2][j] * y + R[r + 3][j] * z) * l;
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
		//print(R);
	}
}