
template<typename T>
void __Schur_HouseHolder_R(MAT<T>& Q, MAT<T>& R, T* v) {
	int m, n;
	T cr, hr;
	T* w;

	for (m = 0; m < R.rows - 2; ++m) {
		w = Q.data + 2 * Q.rows * m - m * m + 1;
		n = m + 1;
		__copy(w, R[m] + n, R.rows - n);
		cr = sqrt(__dot(w, w, R.rows - n));

		if (*w < 0.0)cr = -cr;
		*w += cr;
		hr = 1.0 / sqrt(cr * *w);

		__times(w, hr, w, R.rows - n);

		__ApplyHouseHolderOnTheRight(R, w - n, v, n);
		__ApplyHouseHolderOnTheLeft2(R, w, m, n);
	}

	__HouseHolder_Q2(Q);
}

template<typename T>
inline void __Schur_HouseHolder_Right(T* __restrict a, T* __restrict b, T* __restrict c, const T* __restrict w, int n) {
	T sum;
	for (int i = 0; i < n; ++i) {
		sum = a[i] * w[0] + b[i] * w[1] + c[i] * w[2];
		a[i] -= sum * w[0];
		b[i] -= sum * w[1];
		c[i] -= sum * w[2];
	}
}

template<typename T>
void __Schur_QR_R(MAT<T>& Q, MAT<T>& R, int m, int n, double eps) {
	int i, k, r;
	complex<T> shift;
	T temp, x, y;
	T w[3];
	T* Temp;

	while (m < n) {

		if (fabs(R(m, m + 1)) < eps) {
			++m;
			continue;
		}
		for (r = n - 1; r > m; --r) {
			if (fabs(R(r - 1, r)) < eps) {
				if (n - r <= 2) {
					n = r;
					break;
				}
				else {
					__Schur_QR_R(Q, R, r, n, eps);
					n = r;
					break;
				}
			}
		}

		x = R(n - 2, n - 2) + R(n - 1, n - 1);
		y = x * x - (R(n - 2, n - 2) * R(n - 1, n - 1) - R(n - 2, n - 1) * R(n - 1, n - 2)) * 4;
		if (n - m <= 2) {
			if (n - m < 2) {
				break;
			}
			else if (y < 0) {
				break;
			}
		}

		if (y < 0) {
			w[0] = R(m, m) * R(m, m) + R(m, m + 1) * R(m + 1, m) - x * R(m, m) + (x * x - y) / 4;
			w[1] = R(m, m + 1) * (R(m, m) + R(m + 1, m + 1) - x);
			w[2] = R(m, m + 1) * R(m + 1, m + 2);
			x = w[2] / w[1];
			y = 1.0 / sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
			if (w[0] < 0) y = -y;
			temp = w[1] * y;

			w[1] = sqrt((1 + w[0] * y) / (1 + x * x));
			w[2] = w[1] * x;
			w[0] = -temp / w[1];

			if (m + 4 < n) k = m + 4;
			else k = n;
			for (i = 0; i < k; ++i) {
				Temp = R[i];

				temp = Temp[m] * w[0] + Temp[m + 1] * w[1] + Temp[m + 2] * w[2];
				Temp[m] -= w[0] * temp;
				Temp[m + 1] -= w[1] * temp;
				Temp[m + 2] -= w[2] * temp;
			}
			__Schur_HouseHolder_Right(R[m] + m, R[m + 1] + m, R[m + 2] + m, w, R.cols - m);
			__Schur_HouseHolder_Right(Q[m], Q[m + 1], Q[m + 2], w, Q.rows);

			for (r = m + 1; r < n - 2; ++r) {
				w[0] = R(r - 1, r); w[1] = R(r - 1, r + 1); w[2] = R(r - 1, r + 2);
				temp = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
				if (w[0] < 0.0)temp = -temp;
				w[0] += temp;
				temp = 1.0 / sqrt(temp * w[0]);
				w[0] *= temp; w[1] *= temp; w[2] *= temp;

				if (r + 4 < n) k = r + 4;
				else k = n;
				for (i = 0; i < k; ++i) {
					Temp = R[i];

					temp = Temp[r] * w[0] + Temp[r + 1] * w[1] + Temp[r + 2] * w[2];
					Temp[r] -= w[0] * temp;
					Temp[r + 1] -= w[1] * temp;
					Temp[r + 2] -= w[2] * temp;
				}
				k = r - 1;
				__Schur_HouseHolder_Right(R[r] + k, R[r + 1] + k, R[r + 2] + k, w, R.rows - k);
				__Schur_HouseHolder_Right(Q[r], Q[r + 1], Q[r + 2], w, Q.rows);
			}

			if (fabs(R(r - 1, r + 1)) > fabs(R(r - 1, r))) {
				temp = R(r - 1, r) / R(r - 1, r + 1);
				y = 1 / sqrt(temp * temp + 1);
				x = temp * y;
			}
			else {
				temp = R(r - 1, r + 1) / R(r - 1, r);
				x = 1 / sqrt(temp * temp + 1);
				y = temp * x;
			}

			for (i = 0; i < n; ++i) {
				Temp = R[i];
				temp = Temp[r] * x + Temp[r + 1] * y;
				Temp[r + 1] = Temp[r + 1] * x - Temp[r] * y;
				Temp[r] = temp;
			}
			k = r - 1;
			__ApplyGivensOnTheRight(R[r] + k, R[r + 1] + k, x, y, R.cols - k);
			__ApplyGivensOnTheRight(Q[r], Q[r + 1], x, y, Q.rows);
		}
		else {
			if (x - 2 * R(n - 1, n - 1) > 0) temp = (x - sqrt(y)) * 0.5;
			else temp = (x + sqrt(y)) * 0.5;

			if (fabs(R(m, m + 1)) > fabs(R(m, m) - temp)) {
				temp = (R(m, m) - temp) / R(m, m + 1);
				y = 1 / sqrt(temp * temp + 1);
				x = temp * y;
			}
			else {
				temp = R(m, m + 1) / (R(m, m) - temp);
				x = 1 / sqrt(temp * temp + 1);
				y = temp * x;
			}

			if (m + 3 < n) k = m + 3;
			else k = n;
			for (i = 0; i < k; ++i) {
				Temp = R[i];

				temp = Temp[m] * x + Temp[m + 1] * y;
				Temp[m + 1] = Temp[m + 1] * x - Temp[m] * y;
				Temp[m] = temp;
			}
			__ApplyGivensOnTheRight(R[m] + m, R[m + 1] + m, x, y, R.cols - m);
			__ApplyGivensOnTheRight(Q[m], Q[m + 1], x, y, Q.rows);

			for (r = m + 1; r < n - 1; ++r) {
				if (fabs(R(r - 1, r + 1)) > fabs(R(r - 1, r))) {
					temp = R(r - 1, r) / R(r - 1, r + 1);
					y = 1 / sqrt(temp * temp + 1);
					x = temp * y;
				}
				else {
					temp = R(r - 1, r + 1) / R(r - 1, r);
					x = 1 / sqrt(temp * temp + 1);
					y = temp * x;
				}

				if (r + 3 < n) k = r + 3;
				else k = n;
				for (i = 0; i < k; ++i) {
					Temp = R[i];

					temp = Temp[r] * x + Temp[r + 1] * y;
					Temp[r + 1] = Temp[r + 1] * x - Temp[r] * y;
					Temp[r] = temp;
				}
				k = r - 1;
				__ApplyGivensOnTheRight(R[r] + k, R[r + 1] + k, x, y, R.rows - k);
				__ApplyGivensOnTheRight(Q[r], Q[r + 1], x, y, Q.rows);
			}
		}
	}
}
