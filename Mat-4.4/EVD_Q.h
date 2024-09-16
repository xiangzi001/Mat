
template<typename T>
void EVD_Q_HouseHolder(MAT<T>& Q, MAT<T>& R, T* w) {
	int i, j, k, r;
	T sum, cr, hr;
	for (r = 0; r < R.rows - 1; r++) {
		sum = 0;
		for (k = r + 1; k < R.rows; ++k)
			sum += R[k][r] * R[k][r];
		if (sum == 0) {
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

		//定义Q
		for (i = 0; i < R.rows; ++i)
		{
			sum = 0;
			for (j = r + 1; j < R.rows; ++j)
				sum += Q[i][j] * w[j];
			sum = sum / hr;
			for (j = r + 1; j < R.cols; ++j)
			{
				Q[i][j] -= w[j] * sum;
			}
		}

		//定义R
		for (i = r; i < R.rows; ++i)
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
void EVD_Q_Givens_1(T* &a, T* &b, MAT<T>& Q, int m, int n, double eps) {

	T t, temp, temp_2, c2, s2, cs;
	T B(0);
	int i, k;

	while (n - m >= 0) {

		//double theta = atan2(R[m + 1][m], R[m][m]);
		//T c = cos(theta);
		//T s = sin(theta);

		t = b[m] / (a[m] - a[n + 1]);
		if (fabs(t) < eps) {
			++m;
			continue;
		}
		c2 = 1 / (t * t + 1);
		s2 = 1 - c2;
		cs = t * c2;
		temp = a[m];
		temp_2 = a[m + 1];

		a[m] = c2 * temp + s2 * temp_2 + 2 * cs * b[m];
		a[m + 1] = s2 * temp + c2 * temp_2 - 2 * cs * b[m];
		b[m] = -cs * temp + cs * temp_2 + (c2 - s2) * b[m];
		c2 = sqrt(c2);
		s2 = t * c2;
		B = t * c2 * b[m + 1];
		b[m + 1] = c2 * b[m + 1];

		for (k = 0; k < Q.rows; ++k) {
			temp = c2 * Q[k][m] + s2 * Q[k][m + 1];
			Q[k][m + 1] = -s2 * Q[k][m] + c2 * Q[k][m + 1];
			Q[k][m] = temp;
		}

		for (i = m + 1; i <= n; ++i) {
			t = B / b[i - 1];
			if (fabs(t) < eps) {
				if (n - i) {
					EVD_Q_Givens_1(a, b, Q, m, i - 1, eps);
					m = i + 1;
					break;
				}
				else {
					--n;
					break;
				}
			}
			c2 = 1 / (t * t + 1);
			s2 = 1 - c2;
			cs = t * c2;
			temp = a[i];
			temp_2 = a[i + 1];

			a[i] = c2 * temp + s2 * temp_2 + 2 * cs * b[i];
			a[i + 1] = s2 * temp + c2 * temp_2 - 2 * cs * b[i];
			b[i] = -cs * temp + cs * temp_2 + (c2 - s2) * b[i];
			c2 = sqrt(c2);
			s2 = t * c2;
			b[i - 1] = c2 * b[i - 1] + s2 * B;
			B = s2 * b[i + 1];
			b[i + 1] = c2 * b[i + 1];

			for (k = 0; k < Q.rows; ++k) {
				temp = c2 * Q[k][i] + s2 * Q[k][i + 1];
				Q[k][i + 1] = -s2 * Q[k][i] + c2 * Q[k][i + 1];
				Q[k][i] = temp;
			}
		}
	}
}

template<typename T>
void EVD_Q_QR_1(MAT<T>& A, MAT<T>& Q, double eps) {
	int i;
	int s(Q.rows * Q.cols);
	for (i = 0; i < s; ++i) {
		Q.data[i] = 0;
	}
	for (i = 0; i < Q.rows; ++i) {
		Q[i][i] = 1;
	}

	T* w = new T[A.rows];
	EVD_Q_HouseHolder(Q, A, w);
	delete[] w;

	T* a(A.data);
	T* b(A.data + A.cols);
	for (i = 1; i < A.cols; ++i) {
		a[i] = A[i][i];
	}
	s = A.cols - 1;
	for (i = 1; i < s; ++i) {
		b[i] = A[i + 1][i];
	}
	EVD_Q_Givens_1(a, b, Q, 0, A.rows - 2, eps);
	for (i = 1; i < A.cols; ++i) {
		A[i][i] = a[i];
	}

	for (i = 1; i < A.rows; ++i) {
		A[i - 1][i] = 0;
		A[i][i - 1] = 0;
		//A[i][i + 1] = 0;
	}
	for (i = 2; i < A.cols; ++i) {
		a[i] = 0;
	}
}

template<typename T>
void EVD_Q_QR_test_1(MAT<T>& A, MAT<T>& Q, T* EigenValues, T* b, double eps) {
	int i;
	int s(Q.rows * Q.cols);
	for (i = 0; i < s; ++i) {
		Q.data[i] = 0;
	}
	for (i = 0; i < Q.rows; ++i) {
		Q[i][i] = 1;
	}

	EVD_Q_HouseHolder(Q, A, b);

	for (i = 0; i < A.cols; ++i) {
		EigenValues[i] = A[i][i];
	}
	s = A.cols - 1;
	for (i = 0; i < s; ++i) {
		b[i] = A[i + 1][i];
	}
	EVD_Q_Givens_1(EigenValues, b, Q, 0, A.rows - 2, eps);
}

template<typename T>
void EVD_Q_Eigen_2(T*& a, T*& b, int m, int n, double eps) {

	T t, temp, temp_2, c2, s2, cs;
	T B(0);
	int i;

	while (n - m >= 0) {

		t = b[m] / (a[m] - a[n + 1]);
		if (fabs(t) < eps) {
			++m;
			continue;
		}
		c2 = 1 / (t * t + 1);
		s2 = 1 - c2;
		cs = t * c2;
		temp = a[m];
		temp_2 = a[m + 1];

		a[m] = c2 * temp + s2 * temp_2 + 2 * cs * b[m];
		a[m + 1] = s2 * temp + c2 * temp_2 - 2 * cs * b[m];
		b[m] = -cs * temp + cs * temp_2 + (c2 - s2) * b[m];
		c2 = sqrt(c2);
		s2 = t * c2;
		B = t * c2 * b[m + 1];
		b[m + 1] = c2 * b[m + 1];

		for (i = m + 1; i <= n; ++i) {
			t = B / b[i - 1];
			if (fabs(t) < eps) {
				if (n - i) {
					EVD_Q_Eigen_2(a, b, m, i - 1, eps);
					m = i + 1;
					break;
				}
				else {
					--n;
					break;
				}
			}
			c2 = 1 / (t * t + 1);
			s2 = 1 - c2;
			cs = t * c2;
			temp = a[i];
			temp_2 = a[i + 1];

			a[i] = c2 * temp + s2 * temp_2 + 2 * cs * b[i];
			a[i + 1] = s2 * temp + c2 * temp_2 - 2 * cs * b[i];
			b[i] = -cs * temp + cs * temp_2 + (c2 - s2) * b[i];
			c2 = sqrt(c2);
			s2 = t * c2;
			b[i - 1] = c2 * b[i - 1] + s2 * B;
			B = s2 * b[i + 1];
			b[i + 1] = c2 * b[i + 1];
		}
	}
}

template<typename T>
void EVD_Q_Givens_2(T*& a, T*& b, T*& eigen, MAT<T>& Q) {

	T t, temp, temp_2, c2, s2, cs;
	T B(0);
	int i, k;

	for (int x = Q.rows - 2; x >= 0; --x) {

		//double theta = atan2(R[m + 1][m], R[m][m]);
		//T c = cos(theta);
		//T s = sin(theta);

		t = b[0] / (a[0] - eigen[x+1]);

		c2 = 1 / (t * t + 1);
		s2 = 1 - c2;
		cs = t * c2;
		temp = a[0];
		temp_2 = a[1];

		a[0] = c2 * temp + s2 * temp_2 + 2 * cs * b[0];
		a[1] = s2 * temp + c2 * temp_2 - 2 * cs * b[0];
		b[0] = -cs * temp + cs * temp_2 + (c2 - s2) * b[0];
		c2 = sqrt(c2);
		s2 = t * c2;
		B = t * c2 * b[1];
		b[1] = c2 * b[1];

		for (k = 0; k < Q.rows; ++k) {
			temp = c2 * Q[k][0] + s2 * Q[k][1];
			Q[k][1] = -s2 * Q[k][0] + c2 * Q[k][1];
			Q[k][0] = temp;
		}

		for (i = 1; i <= x; ++i) {
			t = B / b[i - 1];
			c2 = 1 / (t * t + 1);
			s2 = 1 - c2;
			cs = t * c2;
			temp = a[i];
			temp_2 = a[i + 1];

			a[i] = c2 * temp + s2 * temp_2 + 2 * cs * b[i];
			a[i + 1] = s2 * temp + c2 * temp_2 - 2 * cs * b[i];
			b[i] = -cs * temp + cs * temp_2 + (c2 - s2) * b[i];
			c2 = sqrt(c2);
			s2 = t * c2;
			b[i - 1] = c2 * b[i - 1] + s2 * B;
			B = s2 * b[i + 1];
			b[i + 1] = c2 * b[i + 1];

			for (k = 0; k < Q.rows; ++k) {
				temp = c2 * Q[k][i] + s2 * Q[k][i + 1];
				Q[k][i + 1] = -s2 * Q[k][i] + c2 * Q[k][i + 1];
				Q[k][i] = temp;
			}
		}
	}
}
template<typename T>
void EVD_Q_QR_2(MAT<T>& A, MAT<T>& Q, double eps) {
	int i;
	int s(Q.rows* Q.cols);
	T* a(A.data);
	T* b(A[1]);
	T* t_1(A[2]);
	T* t_2(A[3]);


	for (i = 0; i < s; ++i) {
		Q.data[i] = 0;
	}
	for (i = 0; i < Q.rows; ++i) {
		Q[i][i] = 1;
	}

	T* w = new T[A.rows];
	EVD_Q_HouseHolder(Q, A, w);
	delete[] w;

	s = A.rows - 1;
	for (i = 0; i < A.rows; ++i) {
		a[i] = A[i][i];
	}
	for (i = 0; i < s; ++i) {
		b[i] = A[i + 1][i];
	}
	for (i = 0; i < A.rows; ++i) {
		t_1[i] = a[i];
	}
	for (i = 0; i < s; ++i) {
		t_2[i] = b[i];
	}

	EVD_Q_Eigen_2(t_1, t_2, 0, A.rows - 2, eps);
	EVD_Q_Givens_2(a, b, t_1, Q);

	for (i = 0; i < A.rows; ++i) {
		t_1[i] = 0;
	}
	for (i = 1; i < A.rows; ++i) {
		A[i][i] = a[i];
	}
	for (i = 1; i < A.rows; ++i) {
		a[i] = 0;
	}
	for (i = 4; i < A.rows; ++i) {
		A[i - 1][i] = 0;
		A[i][i - 1] = 0;
	}
}

template<typename T>
void EVD_Q_QR_test_2(MAT<T>& A, MAT<T>& Q, T* EigenValues, double eps) {
	int i;
	int s(Q.rows * Q.cols);
	T* t_1	(A.data);
	T* t_2	(A[1]);
	T* b	(A[2]);

	for (i = 0; i < s; ++i) {
		Q.data[i] = 0;
	}
	for (i = 0; i < Q.rows; ++i) {
		Q[i][i] = 1;
	}

	EVD_Q_HouseHolder(Q, A, EigenValues);

	s = A.rows - 1;
	for (i = 0; i < A.rows; ++i) {
		t_1[i] = A[i][i];
	}
	for (i = 0; i < s; ++i) {
		t_2[i] = A[i + 1][i];
	}
	for (i = 0; i < s; ++i) {
		b[i] = t_2[i];
	}
	for (i = 0; i < A.rows; ++i) {
		EigenValues[i] = t_1[i];
	}

	EVD_Q_Eigen_2(t_1, t_2, 0, A.rows - 2, eps);
	EVD_Q_Givens_2(EigenValues, b, t_1, Q);

	for (i = 0; i < A.rows; ++i) {
		t_1[i] = 0;
	}
	for (i = 3; i < A.rows; ++i) {
		A[i - 1][i] = 0;
		A[i][i - 1] = 0;
	}
}


template <typename T>
void EVD_Q_Jacobi(MAT<T>& A, MAT<T>& Q, T dbEps)
{
	int i, j;
	int m, n;
	int k(Q.rows * Q.cols);
	for (i = 0; i < k; ++i) {
		Q.data[i] = 0;
	}
	for (i = 0; i < Q.rows; ++i) {
		Q[i][i] = 1;
	}

	T Max, temp;
	T dbApp, dbApq, dbAqq;
	double angle;
	T s, c, s2, c2;

	// 第二步：对矩阵A.data即矩阵A，求出最大索引m、n即p、q

	while (1)
	{
		// 第二步：对矩阵A.data即矩阵A，求出最大索引m、n即p、q
		Max = A.data[1];
		m = 0;
		n = 1; // 因为求的是非对角线上最大值，所有从1开始

		for (i = 1; i < A.rows; ++i)			//行
		{
			for (j = 0; j < i; ++j)		//列
			{
				temp = fabs(A[i][j]);

				if (temp > Max)
				{
					Max = temp;
					m = i;
					n = j;
				}
			}
		}

		if (Max < dbEps)
			break;

		dbApp = A[m][m];
		dbApq = A[m][n];
		dbAqq = A[n][n];

		angle = 0.5 * atan2(-2 * dbApq, dbAqq - dbApp);

		s = sin(angle);
		c = cos(angle);
		s2 = sin(2 * angle);
		c2 = cos(2 * angle);

		A[m][m] = dbApp * c * c +
			dbAqq * s * s + 2 * dbApq * c * s;
		A[n][n] = dbApp * s * s +
			dbAqq * c * c - 2 * dbApq * c * s;
		A[m][n] = 0.5 * (dbAqq - dbApp) * s2 + dbApq * c2;
		A[n][m] = A[m][n];

		for (i = 0; i < A.rows; ++i)
		{
			if ((i != n) && (i != m))
			{
				temp = A[i][m];
				A[i][m] = A[i][n] * s + temp * c;
				A[i][n] = A[i][n] * c - temp * s;
			}
		}

		for (j = 0; j < A.rows; ++j)
		{
			if ((j != n) && (j != m))
			{
				temp = A[m][j];
				A[m][j] = A[n][j] * s + temp * c;
				A[n][j] = A[n][j] * c - temp * s;
			}
		}

		for (i = 0; i < Q.rows; ++i)
		{
			temp = Q[i][m];
			Q[i][m] = Q[i][n] * s + temp * c;
			Q[i][n] = Q[i][n] * c - temp * s;
		}

	}
	/*
	for (i = 0; i < A.rows; ++i) {
		T m = 0;
		int c = i;
		T value;
		for (j = i; j < A.rows; ++j) {
			if (A[j][j] > m) {
				m = A[j][j];
				c = j;
			}
		}
		if (c != i) {
			value = A[c][c];
			A[c][c] = A[i][i];
			A[i][i] = value;
			for (j = 0; j < A.rows; ++j) {
				value = Q[c][j];
				Q[c][j] = Q[i][j];
				Q[i][j] = value;
			}
		}
	}
	*/
}


template<typename T>
void EVD_Q(MAT<T>& A, MAT<T>& E, MAT<T>& Q, double eps, EVD_type type) {
	switch (type)
	{
	case _EVD_QR_1: {
		for (int i = 0; i < E.rows * E.cols; ++i) {
			E.data[i] = A.data[i];
		}
		EVD_Q_QR_1(E, Q, eps);
		break;
	}
	case _EVD_QR_2: {
		for (int i = 0; i < E.rows * E.cols; ++i) {
			E.data[i] = A.data[i];
		}
		EVD_Q_QR_2(E, Q, eps);
		break;
	}
	case _EVD_Jacobi: {
		for (int i = 0; i < E.rows * E.cols; ++i) {
			E.data[i] = A.data[i];
		}
		EVD_Q_Jacobi(E, Q, eps);
		break;
	}
	default:
		break;
	}
}
