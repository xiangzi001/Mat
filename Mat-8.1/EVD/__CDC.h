
template <class T>
void quickSort(T* arr, int left, int right, int* pos) {
	if (left >= right) {
		return;
	}

	int i = left;
	int j = right;
	T Temp = arr[left];
	int temp = pos[left];
	while (i < j) {
		while (i < j && arr[j] >= Temp) {
			j--;
		}
		arr[i] = arr[j];
		pos[i] = pos[j];

		while (i < j && arr[i] <= Temp) {
			i++;
		}
		arr[j] = arr[i];
		pos[j] = pos[i];
	}
	arr[i] = Temp;
	pos[i] = temp;

	quickSort(arr, left, i - 1, pos);
	quickSort(arr, i + 1, right, pos);
}

#ifdef XiangZiMatImport_SIMD

inline double __EVD_Q_CDC_Solve_EigenValue_Zero(int n, double* z, double* d) {
	__m512d ans(_mm512_setzero_pd());
	int l = n / 8;
	int j;
	for (j = 0; j < l; ++j) {
		ans = _mm512_add_pd(_mm512_div_pd(((__m512d*)z)[j], ((__m512d*)d)[j]), ans);
	}
	l = n - l * 8;
	if (l > 0) {
		ans = _mm512_mask_blend_pd((1 << l) - 1, ans, _mm512_add_pd(_mm512_div_pd(((__m512d*)z)[j], ((__m512d*)d)[j]), ans));
	}
	return _mm512_reduce_add_pd(ans);
}

inline double __EVD_Q_CDC_Solve_EigenValue__(int n, double* z, double* d, double x, int k) {
	__m512d ans(_mm512_setzero_pd());
	__m512d X(_mm512_set1_pd(x));
	int j;

	int l = k / 8;
	for (j = 0; j < l; ++j) {
		ans = _mm512_add_pd(_mm512_div_pd(((__m512d*)z)[j], _mm512_sub_pd(((__m512d*)d)[j], X)), ans);
	}
	l = k - l * 8;
	if (l > 0) {
		ans = _mm512_mask_blend_pd((1 << l) - 1, ans, _mm512_add_pd(_mm512_div_pd(((__m512d*)z)[j], _mm512_sub_pd(((__m512d*)d)[j], X)), ans));
	}

	k += 1;
	z += k;
	d += k;
	n -= k;

	l = n / 8;
	for (j = 0; j < l; ++j) {
		ans = _mm512_add_pd(_mm512_div_pd(((__m512d*)z)[j], _mm512_sub_pd(((__m512d*)d)[j], X)), ans);
	}
	l = n - l * 8;
	if (l > 0) {
		ans = _mm512_mask_blend_pd((1 << l) - 1, ans, _mm512_add_pd(_mm512_div_pd(((__m512d*)z)[j], _mm512_sub_pd(((__m512d*)d)[j], X)), ans));
	}

	return _mm512_reduce_add_pd(ans);
}
inline double __EVD_Q_CDC_Solve_EigenValue__2(int n, double* z, double* d, double x, int k) {
	__m512d ans(_mm512_setzero_pd());
	__m512d X(_mm512_set1_pd(x));
	__m512d temp;
	int j;

	int l = k / 8;
	for (j = 0; j < l; ++j) {
		temp = _mm512_sub_pd(((__m512d*)d)[j], X);
		ans = _mm512_add_pd(_mm512_div_pd(((__m512d*)z)[j], _mm512_mul_pd(temp, temp)), ans);
	}
	l = k - l * 8;
	if (l > 0) {
		temp = _mm512_sub_pd(((__m512d*)d)[j], X);
		ans = _mm512_mask_blend_pd((1 << l) - 1, ans, _mm512_add_pd(_mm512_div_pd(((__m512d*)z)[j], _mm512_mul_pd(temp, temp)), ans));
	}

	k += 1;
	z += k;
	d += k;
	n -= k;

	l = n / 8;
	for (j = 0; j < l; ++j) {
		temp = _mm512_sub_pd(((__m512d*)d)[j], X);
		ans = _mm512_add_pd(_mm512_div_pd(((__m512d*)z)[j], _mm512_mul_pd(temp, temp)), ans);
	}
	l = n - l * 8;
	if (l > 0) {
		temp = _mm512_sub_pd(((__m512d*)d)[j], X);
		ans = _mm512_mask_blend_pd((1 << l) - 1, ans, _mm512_add_pd(_mm512_div_pd(((__m512d*)z)[j], _mm512_mul_pd(temp, temp)), ans));
	}

	return _mm512_reduce_add_pd(ans);
}

inline void __EVD_Q_CDC_u2(int n, double v, double* d, double* u, int k) {

	u[k] *= (v - d[k]);

	int j;
	__m512d temp1(_mm512_set1_pd(v)), temp2(_mm512_set1_pd(d[k]));

	int l = k / 8;
	for (j = 0; j < l; ++j) {
		((__m512d*)u)[j] = _mm512_mul_pd(((__m512d*)u)[j], _mm512_div_pd(_mm512_sub_pd(temp1, ((__m512d*)d)[j]), _mm512_sub_pd(temp2, ((__m512d*)d)[j])));
	}
	l = k - l * 8;
	if (l > 0) {
		((__m512d*)u)[j] = _mm512_mask_blend_pd((1 << l) - 1, ((__m512d*)u)[j], 
			_mm512_mul_pd(((__m512d*)u)[j], _mm512_div_pd(_mm512_sub_pd(temp1, ((__m512d*)d)[j]), _mm512_sub_pd(temp2, ((__m512d*)d)[j])))
		);
	}

	k += 1;
	u += k;
	d += k;
	n -= k;

	l = n / 8;
	for (j = 0; j < l; ++j) {
		((__m512d*)u)[j] = _mm512_mul_pd(((__m512d*)u)[j], _mm512_div_pd(_mm512_sub_pd(temp1, ((__m512d*)d)[j]), _mm512_sub_pd(temp2, ((__m512d*)d)[j])));
	}
	l = n - l * 8;
	if (l > 0) {
		((__m512d*)u)[j] = _mm512_mask_blend_pd((1 << l) - 1, ((__m512d*)u)[j],
			_mm512_mul_pd(((__m512d*)u)[j], _mm512_div_pd(_mm512_sub_pd(temp1, ((__m512d*)d)[j]), _mm512_sub_pd(temp2, ((__m512d*)d)[j])))
		);
	}
}

inline void __EVD_Q_CDC_v(int n, double* u, double* d, double* v, double x) {
	int j;
	__m512d X(_mm512_set1_pd(x));

	int l = n / 8;
	for (j = 0; j < l; ++j) {
		((__m512d*)v)[j] = _mm512_div_pd(((__m512d*)u)[j], _mm512_sub_pd(((__m512d*)d)[j], X));
	}
	l = n - l * 8;
	if (l > 0) {
		((__m512d*)v)[j] = _mm512_mask_blend_pd((1 << l) - 1, ((__m512d*)v)[j],
			_mm512_div_pd(((__m512d*)u)[j], _mm512_sub_pd(((__m512d*)d)[j], X))
		);
	}

}
#endif // XiangZiMatImport_SIMD

template<typename T>
inline T __EVD_Q_CDC_Solve_EigenValue_Zero(int n, T* z, T* d) {
	T a = 0;
	for (int j = 0; j < n; ++j) {
		a += z[j] / d[j];
	}
	return a;
}
template <typename T>
inline double __EVD_Q_CDC_Solve_EigenValue__(int n, T* z, T* d, T x, int k) {
	T a = 0;
	int j;
	for (j = 0; j < k; ++j) {
		a += z[j] / (d[j] - x);
	}
	for (j = k + 1; j < n; ++j) {
		a += z[j] / (d[j] - x);
	}
	return a;
}
template <typename T>
inline double __EVD_Q_CDC_Solve_EigenValue__2(int n, T* z, T* d, T x, int k) {
	int j;
	T b = 0, temp;
	for (j = 0; j < k; ++j) {
		temp = d[j] - x;
		b += z[j] / (temp * temp);
	}
	for (j = k + 1; j < n; ++j) {
		temp = d[j] - x;
		b += z[j] / (temp * temp);
	}
	return b;
}
template<typename T>
inline void __EVD_Q_CDC_u2(int n, T v, T* d, T* u, int k) {
	int i;
	for (i = 0; i < k; ++i) {
		u[i] *= (v - d[i]) / (d[k] - d[i]);
	}
	u[k] *= (v - d[k]);
	for (i = k + 1; i < n; ++i) {
		u[i] *= (v - d[i]) / (d[k] - d[i]);
	}
}
template<typename T>
inline void __EVD_Q_CDC_v(int n, T* u, T* d, T* v, T x) {
	for (int j = 0; j < n; ++j) {
		v[j] = u[j] / (d[j] - x);
	}
}



template<typename T>
inline bool __EVD_Q_CDC_Solve_EigenValue_if(int n, T* z, T* d, int i, T x) {
	return (1 + __EVD_Q_CDC_Solve_EigenValue__(n, z, d, x, n)) > 0;
}

template<typename T>
inline T __EVD_Q_CDC_Solve_EigenValue(int n, T* z, T* d, int i, int k, T x) {
	T a = 1 + __EVD_Q_CDC_Solve_EigenValue__(n, z, d, x, k);
	T b = __EVD_Q_CDC_Solve_EigenValue__2(n, z, d, x, k);

	T c, s, S;
	if (k == i) {
		int j = k + 1;
		c = a + (x - d[j]) * b;
		s = z[k];
		S = (x - d[j]) * (x - d[j]) * b;
	}
	else {
		c = a + (x - d[i]) * b;
		s = (x - d[i]) * (x - d[i]) * b;
		S = z[k];
	}

	a = (d[i] + d[i + 1]) * c + s + S;
	b = c * d[i] * d[i + 1] + s * d[i + 1] + S * d[i];

	if (a > 0)
		return (2 * b) / (a + sqrt(a * a - 4 * b * c));
	else
		return (a - sqrt(a * a - 4 * b * c)) / (2 * c);
}

template<typename T>
inline T __EVD_Q_CDC_Solve_EigenValue2(int n, T* z, T* d, T x) {
	T a = 1 + __EVD_Q_CDC_Solve_EigenValue__(n, z, d, x, n);
	T b = __EVD_Q_CDC_Solve_EigenValue__2(n, z, d, x, n);

	if (a < 0) {
		return x - a / b;
	}

	return x - (a * x) / (a + b * x);
}

template<typename T>
inline void __EVD_Q_CDC_u1(int n, T* u) {
	for (int i = 0; i < n; ++i) {
		if (u[i] < 0)
			u[i] = -1;
		else
			u[i] = 1;
	}
}

template<typename T>
inline void __EVD_Q_CDC_u3(int n, T* u) {
	int i;

	for (i = 0; i < n; ++i) {
		if (u[i] < 0)
			u[i] = -sqrt(-u[i]);
		else
			u[i] = sqrt(u[i]);
	}
}

template<typename t1, typename t2, typename t3>
void times_A_B2(const MAT<t1>& A, const MAT<t2>& B, const MAT<t3>& C, int r1) {
	__BlockMat<t1> _A;
	__BlockMat<t2> _B;
	__BlockMat<t3> _C;
	int i, j, k;
	int l, m, n;
	for (k = 0; k < A.cols; k = k + __BlockSize_times) {
		if (k + __BlockSize_times > A.cols) l = A.cols - k;
		else l = __BlockSize_times;

		for (j = 0; j < B.cols; j = j + __BlockSize_times) {
			if (j + __BlockSize_times > B.cols) n = B.cols - j;
			else n = __BlockSize_times;

			for (i = 0; i < A.rows; i = i + __BlockSize_times) {
				if (i + __BlockSize_times > A.rows) m = A.rows - i;
				else m = __BlockSize_times;

				_A.set(A, i, k, m, l);
				_B.set(B, k, j, l, n);
				_C.set(C, i + r1, j, m, n);
				__times_A_B(_A, _B, _C);
			}
		}
	}
}

template<typename T>
void __EVD_Q_CDC_Solve(T* w, T* v, MAT<T>& Q, T* Temp, T* Temp2, int* pos, T eps) {
	if (Q.rows <= 128) {
		__set0(Q.data, Q.rows * Q.cols);
		for (int i = 0; i < Q.rows; ++i) {
			Q[i][i] = 1;
		}

		__EVD_Q_Givens(w, v, Q, 0, Q.rows - 1, eps);
	}
	else {
		int s = Q.rows / 2;
		int t = Q.rows - s;

		MAT<T> Q1, Q2;
		Q1.rows = s;
		Q1.cols = s;
		Q1.data = Temp;
		Q2.rows = t;
		Q2.cols = t;
		Q2.data = Temp + s * s;

		T p = v[s - 1];
		v[s - 1] = 0;
		w[s - 1] -= p;
		w[s] -= p;

		__EVD_Q_CDC_Solve(w, v, Q1, Temp + t * t + s * s, Temp2, pos, eps);
		__EVD_Q_CDC_Solve(w + s, v + s, Q2, Temp + t * t + s * s, Temp2, pos, eps);

		T* u = Temp + t * t + s * s;
		T* _u = u + Q.rows;
		T* _d = _u + Q.rows;
		T* d = _d + Q.rows;

		int i, j, n = Q.rows, k;
		T x, temp;

		__times(w, 1 / p, v, Q.rows);
		__copy(_d, v, Q.rows);
		for (i = 0; i < s; ++i) {
			u[i] = Q1(s - 1, i);
		}
		for (i = 0; i < t; ++i) {
			u[i + s] = Q2(0, i);
		}

		for (i = Q.rows - 1; i >= 0; --i) {
			if (fabs(u[i]) < eps) {
				--n;

				pos[i] = n;
				pos[n] = i;

				_d[i] = v[n];
				_d[n] = v[i];
			}
			else {
				pos[i] = i;
			}
		}
		quickSort(_d, 0, n - 1, pos);
		__copy(d, _d, Q.rows);

		for (i = 0; i < Q.rows; ++i) {
			_u[i] = u[pos[i]];
		}
		for (i = 0; i < Q.rows; ++i) {
			u[i] = _u[i] * _u[i];
		}

		for (i = 1; i < n; ++i) {
			if (fabs(d[i] - d[i - 1]) < eps * fabs(d[i]) * 100) {
				MessageBoxA(NULL, "wrong", "demo", MB_OK);
			}
		}

		// EigenValues

		__EVD_Q_CDC_u1(n, _u);
		for (i = 0; i < n - 1; ++i) {
			temp = (_d[i] + _d[i + 1]) / 2;

			if (__EVD_Q_CDC_Solve_EigenValue_if(n, u, _d, i, temp)) {
				k = i;
			}
			else {
				k = i + 1;
			}
			temp -= _d[k];
			__add(d, -d[k], _d, n);
			x = __EVD_Q_CDC_Solve_EigenValue(n, u, _d, i, k, temp);

			while (fabs(temp - x) > eps) {
				temp = x;
				x = __EVD_Q_CDC_Solve_EigenValue(n, u, _d, i, k, x);
			}
			__EVD_Q_CDC_u2(n, x, _d, _u, i);
			w[i] = x;
		}

		__add(d, -d[i], _d, n);
		temp = u[i] / __EVD_Q_CDC_Solve_EigenValue_Zero(i, u, _d);
		x = __EVD_Q_CDC_Solve_EigenValue2(n, u, _d, temp);
		while (fabs(temp - x) > eps) {
			temp = x;
			x = __EVD_Q_CDC_Solve_EigenValue2(n, u, _d, x);
		}
		__EVD_Q_CDC_u2(n, x, _d, _u, i);
		w[i] = x;

		for (i = n; i < Q.rows; ++i) {
			_u[i] = 0;
			w[i] = 0;
		}

		__EVD_Q_CDC_u3(n, _u);

		for (i = 0; i < Q.rows; ++i) {
			u[pos[i]] = _u[i];
		}

		// EigenVectors
		__set0(Q.data, Q.rows * Q.cols);
		MAT<T> W1, W2;
		W1.rows = s;
		W1.cols = n;
		W1.data = Temp2;
		W2.rows = t;
		W2.cols = n;
		W2.data = Temp2 + s * n;

		T shift;
		__copy(d, v, Q.rows);
		for (i = 0; i < n; ++i) {
			if (w[i] > 0) {
				shift = d[pos[i]];
			}
			else {
				shift = d[pos[i + 1]];
			}
			__add(d, -shift, _d, Q.rows);

			x = w[i];
			w[i] = (shift + x) * p;

			__EVD_Q_CDC_v(Q.rows, u, _d, v, x);
			temp = (T)1 / sqrt(__dot(v, v, Q.rows));
			__times(v, temp, W1[i], s);
			__times(v + s, temp, W2[i], t);
		}
		times_A_B2(Q1, W1, Q, 0);
		times_A_B2(Q2, W2, Q, s);

		for (i = n; i < Q.rows; ++i) {
			j = pos[i];
			w[i] = d[j] * p;
			if (j < s)
				__copy(Q[i], Q1[j], s);
			else
				__copy(Q[i] + s, Q2[j - s], t);
		}

		Q1.data = 0;
		Q2.data = 0;
		W1.data = 0;
		W2.data = 0;
	}
}


template<typename T>
void __EVD_Q_Householder2(MAT<T>& Q, MAT<T>& R, T* v) {
	int m, n;

	T temp;
	T cr, hr;
	T* w, * a = R[R.rows - 1], * b = R[R.rows - 2] + 1;

	for (m = 0; m < R.rows - 2; ++m) {
		w = Q.data + m * (2 * Q.rows - m + 1) / 2;

		temp = *R.data;

		n = R.rows - m - 1;
		__copy(w, R.data + 1, n);
		cr = sqrt(__dot(w, w, n));

		if (*w < 0.0)cr = -cr;
		*w += cr;
		hr = (T)1 / sqrt(cr * *w);

		__times(w, hr, w, n);

		__set0(v, n);
		for (int i = 0; i < n; ++i) {
			__mixed(v + i, R.data + (n + 2) * (i + 1), w[i], v + i, n - i);
		}
		for (int i = 0; i < n; ++i) {
			v[i] += __dot(R.data + (n + 2) * (i + 1) + 1, w + i + 1, n - i - 1);
		}

		hr = __dot(v, w, n);
		for (int i = 0; i < n; ++i) {
			__mixed(R.data + (n + 2) * (i + 1), v + i, -w[i], R.data + (n + 1) * i, n - i);
			__mixed(R.data + (n + 1) * i, w + i, -v[i] + w[i] * hr, R.data + (n + 1) * i, n - i);
		}
		a[m] = temp;
		b[m] = -cr;
	}
	a[R.rows - 2] = R.data[0];
	a[R.rows - 1] = R.data[3];
	b[R.rows - 2] = R.data[1];
}

template<typename T>
void __EVD_Q_Householder_Q(MAT<T>& Q, MAT<T>& R) {
	int m, n, i;

	T* w;

	for (m = R.rows - 3; m >= 0; --m) {
		w = Q.data + m * (2 * Q.rows - m + 1) / 2;
		n = m + 1;

		for (i = 0; i < R.cols; ++i) {
			__mixed(R[i] + n, w, -__dot(w, R[i] + n, R.rows - n), R[i] + n, R.rows - n);
		}
	}
}

template<typename T>
void __EVD_Q_CDC(MAT<T>& A, MAT<T>& Q, T eps) {
	int i;

	MAT<T> Temp(Q.rows, Q.cols);

	T* w = new T[A.rows];
	T* v = new T[A.cols];
	__EVD_Q_Householder(Temp, A, w);

	__copy(w, A[A.cols - 1], A.cols);
	__copy(v, A[A.cols - 2] + 1, A.cols);
	v[A.cols - 1] = 0;

	int* pos = new int[A.cols];

	T* Temp2 = new T[Q.rows * Q.cols];

	__EVD_Q_CDC_Solve(w, v, A, Temp2, Q.data, pos, eps);

	//__EVD_Q_Householder_Q(Temp, Q);
	times_A_B(Temp, A, Q);

	__set0(A.data, A.rows * A.cols);
	for (i = 0; i < A.rows; ++i) {
		A[i][i] = w[i];
	}

	delete[] w;
	delete[] v;
	delete[] pos;
	delete[] Temp2;
}
