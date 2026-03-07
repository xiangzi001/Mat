
template<typename T>
void __EVD_Q_HouseHolder(MAT<T>& Q, MAT<T>& R, T* w, T* v) {
	int m, n;

	T cr, hr;

	for (m = 0; m < R.rows - 2; ++m) {
		n = m + 1;
		__copy(w + n, R[m] + n, R.rows - n);
		cr = sqrt(__dot(w + n, w + n, R.rows - n));

		if (w[n] < 0.0)cr = -cr;
		w[n] += cr;
		hr = 1.0 / sqrt(cr * w[n]);

		__times(w + n, hr, w + n, R.rows - n);

		__ApplyHouseHolderOnTheRight(Q, w, v, n);

		__ApplyHouseHolderOnTheRight2(R, w, v, m, n);
		__ApplyHouseHolderOnTheLeft2(R, w, m, n);
	}
}

template<typename T>
void __EVD_Q_Givens(T* a, T* b, MAT<T>& Q, int m, int n, double eps) {

	T t, temp, temp_2, c2, s2, cs;
	T B(0);
	int i;

	while (n > m) {
		if (fabs(b[m]) < eps) {
			++m;
			continue;
		}
		if (fabs(b[n - 1]) < eps) {
			--n;
			continue;
		}
		for (i = m + 1; i < n - 1; ++i) {
			if (fabs(b[i]) < eps) {
				__EVD_Q_Givens(a, b, Q, m, i, eps);
				m = i + 1;
			}
		}

		c2 = a[n - 1] + a[n];
		s2 = sqrt(c2 * c2 - (a[n - 1] * a[n] - b[n - 1] * b[n - 1]) * 4);
		if (c2 - 2 * a[n] > 0) t = (c2 - s2) * 0.5;
		else t = (c2 + s2) * 0.5;

		if (fabs(b[m]) > fabs(a[m] - t)) {
			t = (a[m] - t) / b[m];
			s2 = 1 / (t * t + 1);
			c2 = 1 - s2;
			cs = t * s2;
			temp = a[m];
			temp_2 = a[m + 1];

			a[m] = c2 * temp + s2 * temp_2 + 2 * cs * b[m];
			a[m + 1] = s2 * temp + c2 * temp_2 - 2 * cs * b[m];
			b[m] = -cs * temp + cs * temp_2 + (c2 - s2) * b[m];
			s2 = sqrt(s2);
			c2 = t * s2;
		}
		else {
			t = b[m] / (a[m] - t);
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
		}
		B = s2 * b[m + 1];
		b[m + 1] = c2 * b[m + 1];

		__ApplyGivensOnTheRight(Q[m], Q[m + 1], c2, s2, Q.rows);

		for (i = m + 1; i < n; ++i) {
			if (fabs(B) > fabs(b[i - 1])) {
				t = b[i - 1] / B;
				s2 = 1 / (t * t + 1);
				c2 = 1 - s2;
				cs = t * s2;
				temp = a[i];
				temp_2 = a[i + 1];

				a[i] = c2 * temp + s2 * temp_2 + 2 * cs * b[i];
				a[i + 1] = s2 * temp + c2 * temp_2 - 2 * cs * b[i];
				b[i] = -cs * temp + cs * temp_2 + (c2 - s2) * b[i];
				s2 = sqrt(s2);
				c2 = t * s2;
			}
			else {
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
			}
			b[i - 1] = c2 * b[i - 1] + s2 * B;
			B = s2 * b[i + 1];
			b[i + 1] = c2 * b[i + 1];

			__ApplyGivensOnTheRight(Q[i], Q[i + 1], c2, s2, Q.rows);
		}
	}
}

template<typename T>
void __EVD_Q_QR(MAT<T>& A, MAT<T>& Q, double eps) {
	int i;
	int s;

	__set0(Q.data, Q.rows * Q.cols);
	for (i = 0; i < Q.rows; ++i) {
		Q[i][i] = 1;
	}

	T* w = new T[A.rows];
	T* v = new T[A.cols];
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
	__EVD_Q_Givens(a, b, Q, 0, A.rows - 1, eps);
	for (i = 1; i < A.cols; ++i) {
		A[i][i] = a[i];
	}

	for (i = 1; i < A.rows; ++i) {
		A[i - 1][i] = 0;
		A[i][i - 1] = 0;
	}
	for (i = 2; i < A.cols; ++i) {
		a[i] = 0;
	}
}
