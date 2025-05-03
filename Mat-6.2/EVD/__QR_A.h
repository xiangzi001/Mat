

template<typename T>
void __EVD_Q_Givens_A(T* a, T* b, T* value, MAT<T>& Q, int m, int n, double eps) {

	T temp, temp_2, c, s, c2, s2, cs;
	T B(0);
	int i, k;
	double angle;

	while (n > m) {
		if (fabs(b[m]) < eps) {
			++m;
			continue;
		}
		for (i = m + 1; i < n - 1; ++i) {
			if (fabs(b[i]) < eps) {
				__EVD_Q_Givens_A(a, b, value, Q, m, i, eps);
				m = i + 1;
			}
		}

		angle = atan2(b[m], a[m] - value[n]);

		c = cos(angle);
		s = sin(angle);

		c2 = c * c;
		s2 = s * s;
		cs = c * s;
		temp = a[m];
		temp_2 = a[m + 1];

		a[m] = c2 * temp + s2 * temp_2 + 2 * cs * b[m];
		a[m + 1] = s2 * temp + c2 * temp_2 - 2 * cs * b[m];
		b[m] = -cs * temp + cs * temp_2 + (c2 - s2) * b[m];

		B = s * b[m + 1];
		b[m + 1] = c * b[m + 1];

		for (k = 0; k < Q.rows; ++k) {
			temp = c * Q[k][m] + s * Q[k][m + 1];
			Q[k][m + 1] = -s * Q[k][m] + c * Q[k][m + 1];
			Q[k][m] = temp;
		}

		for (i = m + 1; i < n; ++i) {
			angle = atan2(B, b[i - 1]);
			c = cos(angle);
			s = sin(angle);

			c2 = c * c;
			s2 = s * s;
			cs = c * s;
			temp = a[i];
			temp_2 = a[i + 1];

			a[i] = c2 * temp + s2 * temp_2 + 2 * cs * b[i];
			a[i + 1] = s2 * temp + c2 * temp_2 - 2 * cs * b[i];
			b[i] = -cs * temp + cs * temp_2 + (c2 - s2) * b[i];

			b[i - 1] = c * b[i - 1] + s * B;
			B = s * b[i + 1];
			b[i + 1] = c * b[i + 1];

			for (k = 0; k < Q.rows; ++k) {
				temp = c * Q[k][i] + s * Q[k][i + 1];
				Q[k][i + 1] = -s * Q[k][i] + c * Q[k][i + 1];
				Q[k][i] = temp;
			}
		}

		if (fabs(b[n - 1]) < eps) {
			--n;
		}
		else {
			value[n] = a[n];
		}
	}
}


template<typename T>
void __EVD_Q_QR_A(MAT<T>& A, MAT<T>& Q, double eps) {
	int i;
	int s(Q.rows * Q.cols);
	T* a(A.data);
	T* b(A[1]);

	ZeroMemory(Q.data, sizeof(T) * Q.rows * Q.cols);
	for (i = 0; i < Q.rows; ++i) {
		Q[i][i] = 1;
	}

	T* w = new T[A.rows + 64 / sizeof(T) - 1];
	T* v = new T[A.cols + 64 / sizeof(T) - 1];
	__EVD_Q_HouseHolder(Q, A, w, v);

	s = A.rows - 1;
	for (i = 0; i < A.rows; ++i) {
		w[i] = A[i][i];
	}
	for (i = 0; i < s; ++i) {
		v[i] = A[i + 1][i];
	}
	v[A.rows - 1] = 0;

	memcpy(a, w, sizeof(T) * A.rows);
	memcpy(b, v, sizeof(T) * A.rows);

	__EVD_Q_Eigen_2(w, v, 0, A.rows - 1, eps);
	__EVD_Q_Givens_A(a, b, w, Q, 0, A.rows - 1, eps);

	for (i = 1; i < A.rows; ++i) {
		A[i][i] = a[i];
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
