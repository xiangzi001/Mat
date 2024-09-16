
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
void __Schur_QR_R(MAT<T>& X, MAT<T>& R, int m, int n, T eps) {
	int i, j, k, r;
	T sum, l, x, y, z;
	complex<T> shift;

	while (n - m > 2) {

		if (fabs(R[m + 1][m]) < eps) {
			++m;
		}

		x = R[n - 2][n - 2] + R[n - 1][n - 1];
		y = x * x - (R[n - 2][n - 2] * R[n - 1][n - 1] - R[n - 2][n - 1] * R[n - 1][n - 2]) * 4;
		if (y < 0 && m - n > 3) {
			shift.real = x * 0.5;
			shift.imag = sqrt(-y) * 0.5;


			x = R[m][m] * R[m][m] + R[m][m + 1] * R[m + 1][m] - 2 * shift.real * R[m][m] + shift.len2();
			y = R[m + 1][m] * (R[m][m] + R[m + 1][m + 1] - 2 * shift.real);
			z = R[m + 1][m] * R[m + 2][m + 1];
			l = 1.0 / sqrt(x * x + y * y + z * z);
			x = sqrt((1 - x * l) * 0.5);
			y = -(y * l / x) * 0.5;
			z = -(z * l / x) * 0.5;

			for (i = 0; i < X.rows; ++i)
			{
				sum = (X[i][m] * x
					+ X[i][m + 1] * y
					+ X[i][m + 2] * z) * 2;
				X[i][m] -= x * sum;
				X[i][m + 1] -= y * sum;
				X[i][m + 2] -= z * sum;
			}
			if (m + 3 < n) k = m + 3;
			else k = n;
			for (i = 0; i < k; ++i)
			{
				sum = (R[i][m] * x
					+ R[i][m + 1] * y
					+ R[i][m + 2] * z) * 2;
				R[i][m] -= x * sum;
				R[i][m + 1] -= y * sum;
				R[i][m + 2] -= z * sum;
			}
			for (j = m; j < R.cols; ++j)
			{
				sum = (R[m][j] * x
					+ R[m + 1][j] * y
					+ R[m + 2][j] * z) * 2;
				R[m][j] -= x * sum;
				R[m + 1][j] -= y * sum;
				R[m + 2][j] -= z * sum;
			}

			for (r = m; r < n - 3; ++r) {
				sum = R[r + 1][r] * R[r + 1][r]
					+ R[r + 2][r] * R[r + 2][r]
					+ R[r + 3][r] * R[r + 3][r];
				l = sqrt(sum);
				if (R[r + 1][r] > 0.0)l = -l;

				//定义w
				x = R[r + 1][r] - l;
				y = R[r + 2][r];
				z = R[r + 3][r];

				l = 1.0 / (l * (l - R[r + 1][r]));
				//定义X
				for (i = 0; i < X.rows; ++i) {
					sum = (X[i][r + 1] * x
						+ X[i][r + 2] * y
						+ X[i][r + 3] * z) * l;
					X[i][r + 1] -= x * sum;
					X[i][r + 2] -= y * sum;
					X[i][r + 3] -= z * sum;
				}
				//定义R
				if (r + 3 < n) k = r + 3;
				else k = n;
				for (i = 0; i < k; ++i) {
					sum = (R[i][r + 1] * x
						+ R[i][r + 2] * y
						+ R[i][r + 3] * z) * l;
					R[i][r + 1] -= x * sum;
					R[i][r + 2] -= y * sum;
					R[i][r + 3] -= z * sum;
				}
				for (j = r; j < R.cols; ++j)
				{
					sum = (R[r + 1][j] * x
						+ R[r + 2][j] * y
						+ R[r + 3][j] * z) * l;
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
			for (i = 0; i < X.rows; ++i) {
				sum = X[i][r + 1] * x + X[i][r + 2] * y;
				X[i][r + 2] = X[i][r + 2] * x - X[i][r + 1] * y;
				X[i][r + 1] = sum;
			}
			//cout << "Householder" << endl;
			//cout << m << ",	" << n << endl;
		}
		else {
			//cout << "Givens" << endl;
			//cout << m << ",	" << n << endl;
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
			for (i = 0; i < X.rows; ++i) {
				sum = X[i][m] * x + X[i][m + 1] * y;
				X[i][m + 1] = X[i][m + 1] * x - X[i][m] * y;
				X[i][m] = sum;
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
				for (i = 0; i < X.rows; ++i) {
					sum = X[i][r + 1] * x + X[i][r + 2] * y;
					X[i][r + 2] = X[i][r + 2] * x - X[i][r + 1] * y;
					X[i][r + 1] = sum;
				}
			}
		}
		//print(R);

		for (r = n - 1; r > 0; --r) {
			if (fabs(R[r][r - 1]) < eps) {
				if (n - r <= 2) {
					n = r;
					break;
				}
				else {
					__Schur_QR_R(X, R, r, n, eps);
					n = r;
					break;
				}
			}
		}
	}
}
