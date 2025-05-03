
template<typename T>
void __EVD_Q_Givens_1(T*& a, T*& b, MAT<T>& Q, int m, int n, double eps) {

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
					__EVD_Q_Givens_1(a, b, Q, m, i - 1, eps);
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
void __EVD_Q_QR_1(MAT<T>& A, MAT<T>& Q, double eps) {
	int i;
	int s(Q.rows * Q.cols);
	for (i = 0; i < s; ++i) {
		Q.data[i] = 0;
	}
	for (i = 0; i < Q.rows; ++i) {
		Q[i][i] = 1;
	}

	T* w = new T[A.rows + 64 / sizeof(T) - 1];
	T* v = new T[A.cols + 64 / sizeof(T) - 1];
	__EVD_Q_HouseHolder(Q, A, w, v);
	delete[] w;
	delete[] v;

	T* a(A.data);
	T* b(A.data + A.cols);
	for (i = 1; i < A.cols; ++i) {
		a[i] = A[i][i];
	}
	s = A.cols - 1;
	for (i = 1; i < s; ++i) {
		b[i] = A[i + 1][i];
	}
	__EVD_Q_Givens_1(a, b, Q, 0, A.rows - 2, eps);
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
void __EVD_Q_QR_test_1(MAT<T>& A, MAT<T>& Q, T* EigenValues, T* b, double eps) {
	int i;
	int s(Q.rows * Q.cols);
	for (i = 0; i < s; ++i) {
		Q.data[i] = 0;
	}
	for (i = 0; i < Q.rows; ++i) {
		Q[i][i] = 1;
	}

	T* v = new T[A.cols + 64 / sizeof(T) - 1];
	__EVD_Q_HouseHolder(Q, A, b, v);
	delete[] v;

	for (i = 0; i < A.cols; ++i) {
		EigenValues[i] = A[i][i];
	}
	s = A.cols - 1;
	for (i = 0; i < s; ++i) {
		b[i] = A[i + 1][i];
	}
	__EVD_Q_Givens_1(EigenValues, b, Q, 0, A.rows - 2, eps);
}
