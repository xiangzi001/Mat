
#ifdef XiangZiMatImport_SIMD
inline void __EVD_Q_ADC_v(int n, double* u, double* d, double* v, double x) {
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

	v[0] = -1;
}

inline void __EVD_Q_ADC_u1(int n, double* u, double* d, double x) {
	int j;
	__m512d X(_mm512_set1_pd(x));

	int l = n / 8;
	for (j = 0; j < l; ++j) {
		((__m512d*)u)[j] = _mm512_mul_pd(((__m512d*)u)[j], _mm512_sub_pd(((__m512d*)d)[j], X));
	}
	l = n - l * 8;
	if (l > 0) {
		((__m512d*)u)[j] = _mm512_mask_blend_pd((1 << l) - 1, ((__m512d*)u)[j],
			_mm512_mul_pd(((__m512d*)u)[j], _mm512_sub_pd(((__m512d*)d)[j], X))
		);
	}
}
#endif // XiangZiMatImport_SIMD

template <typename T>
inline void __EVD_Q_ADC_u1(int n, T* u, T* d, T x) {
	for (int j = 1; j < n; ++j) {
		u[j] *= d[j] - x;
	}
}

template <typename T>
inline void __EVD_Q_ADC_v(int n, T* u, T* d, T* v, T x) {
	v[0] = -1;
	for (int j = 1; j < n; ++j) {
		v[j] = u[j] / (d[j] - x);
	}
}


template<typename T>
inline bool __EVD_Q_ADC_Solve_EigenValue_if(int n, T* z, T* d, int i, T x, T alpha) {
	return(x - alpha + __EVD_Q_CDC_Solve_EigenValue__(n - 1, z + 1, d + 1, x, n)) > 0;
}

template<typename T>
inline T __EVD_Q_ADC_Solve_EigenValue(int n, T* z, T* d, int i, int k, T x, T alpha) {
	int j;

	T a = x - alpha + __EVD_Q_CDC_Solve_EigenValue__(n - 1, z + 1, d + 1, x, k - 1);
	T b = 1 + __EVD_Q_CDC_Solve_EigenValue__2(n - 1, z + 1, d + 1, x, k - 1);

	T c, s, S;
	if (k == i) {
		j = k + 1;
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
inline T __EVD_Q_ADC_Solve_EigenValue1(int n, T* z, T* d, T x, T alpha) {
	T a = x - alpha + __EVD_Q_CDC_Solve_EigenValue__(n - 1, z + 1, d + 1, x, n - 1);
	T b = 1 + __EVD_Q_CDC_Solve_EigenValue__2(n - 1, z + 1, d + 1, x, n - 1);

	if (a > 0) {
		return x - a / b;
	}

	T v;
	v = d[1] - x;

	return x + (a * v) / (a - b * v);
}

template<typename T>
inline T __EVD_Q_ADC_Solve_EigenValue2(int n, T* z, T* d, T x, T alpha) {
	T a = x - alpha + __EVD_Q_CDC_Solve_EigenValue__(n - 1, z + 1, d + 1, x, n - 1);
	T b = 1 + __EVD_Q_CDC_Solve_EigenValue__2(n - 1, z + 1, d + 1, x, n - 1);

	if (a < 0) {
		return x - a / b;
	}

	T v;
	v = d[n - 1] - x;

	return x + (a * v) / (a - b * v);
}

template<typename T>
void __EVD_Q_ADC_Solve(T* w, T* v, MAT<T>& Q, T* Temp, T* Temp2, int* pos, T eps) {
	if (Q.rows <= 128) {
		__set0(Q.data, Q.rows * Q.cols);
		for (int i = 0; i < Q.rows; ++i) {
			Q[i][i] = 1;
		}

		__EVD_Q_Givens(w, v, Q, 0, Q.rows - 1, eps);
	}
	else {
		int s = Q.rows / 2;
		int t = Q.rows - 1 - s;

		MAT<T> Q1, Q2;
		Q1.rows = s;
		Q1.cols = s;
		Q1.data = Temp;
		Q2.rows = t;
		Q2.cols = t;
		Q2.data = Temp + s * s;

		T alpha = w[s];
		T Beta1 = v[s - 1];
		T Beta2 = v[s];
		v[s - 1] = 0;
		v[s] = 0;
		w[s] = 0;

		__EVD_Q_ADC_Solve(w, v, Q1, Temp + s * s + t * t, Temp2, pos, eps);
		__EVD_Q_ADC_Solve(w + s + 1, v + s + 1, Q2, Temp + s * s + t * t, Temp2, pos, eps);

		T* u = Temp + t * t + s * s;
		T* _u = u + Q.rows;
		T* _d = _u + Q.rows;
		T* d = _d + Q.rows;

		int i, j, n = Q.rows, k;
		T x, temp, shift;

		v[0] = 0;
		__copy(v + 1, w, s);
		__copy(v + s + 1, w + s + 1, t);
		__copy(_d, v, Q.rows);

		for (i = 0; i < s; ++i) {
			u[i + 1] = Q1(s - 1, i);
		}
		__times(u + 1, Beta1, u + 1, s);
		for (i = 0; i < t; ++i) {
			u[i + s + 1] = Q2(0, i);
		}
		__times(u + s + 1, Beta2, u + s + 1, t);

		for (i = n - 1; i > 0; --i) {
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
		quickSort(_d, 1, n - 1, pos);
		__copy(d, _d, Q.rows);

		_u[0] = 0;
		for (i = 1; i < Q.rows; ++i) {
			_u[i] = u[pos[i]];
		}
		for (i = 1; i < Q.rows; ++i) {
			u[i] = _u[i] * _u[i];
		}
		__EVD_Q_CDC_u1(Q.rows, _u);

		// EigenValues

		T flag = eps * fabs(d[1] - d[n - 1]);

		shift = alpha - d[1];
		__add(d, -d[1], _d, n);
		temp = __EVD_Q_CDC_Solve_EigenValue_Zero(n - 2, u + 2, _d + 2) - shift;
		if (temp < 0) temp = u[1] / temp;
		else temp = -temp;

		x = __EVD_Q_ADC_Solve_EigenValue1(n, u, _d, temp, shift);
		while (fabs(temp - x) > flag) {
			temp = x;
			x = __EVD_Q_ADC_Solve_EigenValue1(n, u, _d, x, shift);
		}
		w[0] = x;
		__EVD_Q_ADC_u1(Q.rows, _u, _d, x);

		for (i = 1; i < n - 1; ++i) {
			temp = (_d[i] + _d[i + 1]) / 2;
			if (__EVD_Q_ADC_Solve_EigenValue_if(n, u, _d, i, temp, shift)) {
				k = i;
			}
			else {
				k = i + 1;
			}
			temp -= _d[k];
			shift = alpha - d[k];
			__add(d, -d[k], _d, n);

			x = __EVD_Q_ADC_Solve_EigenValue(n, u, _d, i, k, temp, shift);
			while (fabs(temp - x) > flag) {
				temp = x;
				x = __EVD_Q_ADC_Solve_EigenValue(n, u, _d, i, k, x, shift);
			}
			w[i] = x;
			__EVD_Q_CDC_u2(Q.rows, x, _d, _u, i);
		}

		shift = alpha - d[i];
		__add(d, -d[i], _d, n);
		temp = __EVD_Q_CDC_Solve_EigenValue_Zero(i - 1, u + 1, _d + 1) - shift;
		if (temp > 0) temp = u[i] / temp;
		else temp = -temp;

		x = __EVD_Q_ADC_Solve_EigenValue2(n, u, _d, temp, shift);
		while (fabs(temp - x) > flag) {
			temp = x;
			x = __EVD_Q_ADC_Solve_EigenValue2(n, u, _d, x, shift);
		}
		w[i] = x;
		__EVD_Q_CDC_u2(Q.rows, x, _d, _u, i);

		for (i = n; i < Q.rows; ++i) {
			_u[i] = 0;
			w[i] = 0;
		}

		__add(d, -d[1], _d, n);
		__EVD_Q_CDC_u3(n, _u);

		for (i = 1; i < Q.rows; ++i) {
			u[pos[i]] = _u[i];
		}

		/*
		for (i = 0; i < Q.rows; ++i) {
			std::cout << w[i] << std::endl;
		}
		std::cout << std::endl << std::endl << std::endl << std::endl << std::endl << std::endl << std::endl;
		//*/

		// EigenVectors
		__set0(Q.data, Q.rows * Q.cols);

		MAT<T> W1, W2;
		W1.rows = s;
		W1.cols = n;
		W1.data = Temp2;
		W2.rows = t;
		W2.cols = n;
		W2.data = Temp2 + s * n;

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
			w[i] = shift + x;

			__EVD_Q_ADC_v(Q.rows, u, _d, v, x);
			temp = 1 / sqrt(__dot(v, v, Q.rows));

			__times(v + 1, temp, W1[i], s);
			__times(v + s + 1, temp, W2[i], t);
			Q(s, i) = v[0] * temp;
		}
		times_A_B2(Q1, W1, Q, 0);
		times_A_B2(Q2, W2, Q, s + 1);

		for (i = n; i < Q.rows; ++i) {
			j = pos[i];
			w[i] = d[j];
			if (j <= s)
				__copy(Q[i], Q1[j - 1], s);
			else
				__copy(Q[i] + s + 1, Q2[j - s - 1], t);
		}

		Q1.data = 0;
		Q2.data = 0;
		W1.data = 0;
		W2.data = 0;
	}
}


template<typename T>
void __EVD_Q_ADC(MAT<T>& A, MAT<T>& Q, T eps) {
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

	__EVD_Q_ADC_Solve(w, v, A, Temp2, Q.data, pos, eps);

	times_A_B(Temp, A, Q);
	//__EVD_Q_Householder_Q(Temp, Q);

	__set0(A.data, A.rows * A.cols);
	for (i = 0; i < A.rows; ++i) {
		A[i][i] = w[i];
	}

	delete[] w;
	delete[] v;
	delete[] pos;
}
