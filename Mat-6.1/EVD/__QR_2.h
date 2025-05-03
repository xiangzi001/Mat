
template<typename T>
void __EVD_Q_Eigen_2(T*& a, T*& b, int m, int n, double eps) {

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
					__EVD_Q_Eigen_2(a, b, m, i - 1, eps);
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
void __EVD_Q_Givens_2(T*& a, T*& b, T*& eigen, MAT<T>& Q) {

	T t, temp, temp_2, c2, s2, cs;
	T B(0);
	int i, k;

	for (int x = Q.rows - 2; x >= 0; --x) {

		//double theta = atan2(R[m + 1][m], R[m][m]);
		//T c = cos(theta);
		//T s = sin(theta);

		t = b[0] / (a[0] - eigen[x + 1]);

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
void __EVD_Q_QR_2(MAT<T>& A, MAT<T>& Q, double eps) {
	int i;
	int s(Q.rows * Q.cols);
	T* a(A.data);
	T* b(A[1]);


	for (i = 0; i < s; ++i) {
		Q.data[i] = 0;
	}
	for (i = 0; i < Q.rows; ++i) {
		Q[i][i] = 1;
	}

	T* w = new T[A.rows + 64 / sizeof(T) - 1];
	T* v = new T[A.cols + 64 / sizeof(T) - 1];
	__EVD_Q_HouseHolder(Q, A, w, v);

	s = A.rows - 1;
	for (i = 0; i < A.rows; ++i) {
		a[i] = A[i][i];
	}
	for (i = 0; i < s; ++i) {
		b[i] = A[i + 1][i];
	}
	for (i = 0; i < A.rows; ++i) {
		w[i] = a[i];
	}
	for (i = 0; i < s; ++i) {
		v[i] = b[i];
	}

	__EVD_Q_Eigen_2(a, b, 0, A.rows - 2, eps);
	__EVD_Q_Givens_2(w, v, a, Q);

	for (i = 1; i < A.rows; ++i) {
		A[i][i] = w[i];
	}
	delete[] w;
	delete[] v;

	for (i = 1; i < A.rows; ++i) {
		a[i] = 0;
	}
	for (i = 1; i < A.rows; ++i) {
		A[i - 1][i] = 0;
		A[i][i - 1] = 0;
	}
}

template<typename T>
void __EVD_Q_QR_test_2(MAT<T>& A, MAT<T>& Q, T* EigenValues, double eps) {
	int i;
	int s(Q.rows * Q.cols);
	T* t_1(A.data);
	T* t_2(A[1]);
	T* b(A[2]);

	for (i = 0; i < s; ++i) {
		Q.data[i] = 0;
	}
	for (i = 0; i < Q.rows; ++i) {
		Q[i][i] = 1;
	}

	T* v = new T[A.cols + 64 / sizeof(T) - 1];
	__EVD_Q_HouseHolder(Q, A, EigenValues, v);
	delete[] v;

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

	__EVD_Q_Eigen_2(t_1, t_2, 0, A.rows - 2, eps);
	__EVD_Q_Givens_2(EigenValues, b, t_1, Q);

	for (i = 0; i < A.rows; ++i) {
		t_1[i] = 0;
	}
	for (i = 3; i < A.rows; ++i) {
		A[i - 1][i] = 0;
		A[i][i - 1] = 0;
	}
}