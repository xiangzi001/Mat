
template<typename T>
void __LQ_HouseHolder(MAT<T>& Q, MAT<T>& L, T* w) {
	int n = 0;
	if (L.cols > L.rows) {
		n = L.rows;
	}
	else {
		n = L.cols;
	};
	int i, j, k, r;

	T sum, cr, hr;

	for (r = 0; r < n; ++r) {

		sum = 0;
		for (k = r; k < L.cols; ++k)
			sum += L[r][k] * L[r][k];
		cr = sqrt(sum);

		if (L[r][r] < 0.0)cr = -cr;
		hr = 1 / (cr * (cr + L[r][r]));

		//定义w
		w[r] = L[r][r] + cr;
		for (i = r + 1; i < L.cols; ++i)
		{
			w[i] = L[r][i];
		}

		for (i = 0; i < L.cols; ++i)//定义Q
		{
			sum = 0;
			for (j = r; j < n; ++j)
				sum += Q[i][j] * w[j];
			sum = sum * hr;
			for (j = r; j < L.rows; ++j)
			{
				Q[i][j] = Q[i][j] - w[j] * sum;
			}
		}
		for (j = 0; j < L.rows; ++j)//定义L
		{
			sum = 0;
			for (i = r; i < n; ++i)
				sum += L[j][i] * w[i];
			sum = sum * hr;
			for (i = r; i < L.cols; ++i)
				L[j][i] = L[j][i] - w[i] * sum;
		}

	}
}

template <typename T>
bool __SVD_Jacobi_Rot(MAT<T>& U, MAT<T>& V, int i, int j, T eps) {
	int k = 0;

	T a = 0;
	for (k = 0; k < V.cols; ++k) {
		a = a + V[k][i] * V[k][j];
	}
	if (a * a < eps) {
		return true;
	}
	T b = 0;
	for (k = 0; k < V.cols; ++k) {
		b = b - V[k][i] * V[k][i] + V[k][j] * V[k][j];
	}


	double Angle = 0.5 * atan2(2 * a, b);
	T c = cos(Angle);
	T s = sin(Angle);

	for (k = 0; k < V.cols; ++k) {
		a = V[k][i] * c - V[k][j] * s;
		b = V[k][i] * s + V[k][j] * c;
		V[k][i] = a;
		V[k][j] = b;
	}
	for (k = 0; k < U.rows; ++k) {
		a = U[k][i] * c - U[k][j] * s;
		b = U[k][i] * s + U[k][j] * c;
		U[k][i] = a;
		U[k][j] = b;
	}

	return false;

}

template <typename T>
void __SVD_Jacobi(MAT<T>& A, MAT<T>& U, MAT<T>& S, MAT<T>& V, double eps) {
	int i, j;
	int s = S.rows * S.cols;
	for (i = 0; i < s; ++i) {
		S.data[i] = A.data[i];
	}

	if (A.rows > A.cols) {
		int m = 0, n = S.cols - 1;

		s = U.rows * U.cols;
		for (i = 0; i < s; ++i) {
			U.data[i] = 0;
		}
		for (i = 0; i < U.rows; ++i) {
			U[i][i] = 1;
		}

		T* w = new T[A.cols];
		__QR_HouseHolder(U, S, w);

		for (i = 0; i < S.cols; ++i) {
			for (j = 0; j <= i; ++j) {
				V[i][j] = S[j][i];
				S[j][i] = 0;
			}
			for (j = i + 1; j < S.cols; ++j) {
				V[i][j] = 0;
			}
		}

		bool flag = false;
		while (m < n) {
			flag = true;
			for (j = m + 1; j <= n; ++j) {
				flag = flag && __SVD_Jacobi_Rot(U, V, m, j, eps);
			}
			if (flag) {
				++m;
				continue;
			}

			flag = true;
			for (j = m; j < n; ++j) {
				flag = flag && __SVD_Jacobi_Rot(U, V, j, n, eps);
			}
			if (flag) {
				--n;
				continue;
			}

			for (i = m + 1; i < n; ++i) {
				for (j = i + 1; j <= n; ++j) {
					__SVD_Jacobi_Rot(U, V, i, j, eps);
				}
			}
		}

		for (i = 0; i < V.cols; ++i) {
			w[i] = 0;
		}
		for (i = 0; i < V.cols; ++i) {
			for (j = 0; j < V.cols; ++j) {
				w[j] += V[i][j] * V[i][j];
			}
		}
		for (i = 0; i < S.cols; ++i) {
			S[i][i] = sqrt(w[i]);
			w[i] = 1 / S[i][i];
		}
		for (i = 0; i < V.cols; ++i) {
			for (j = 0; j < V.rows; ++j) {
				V[i][j] = V[i][j] * w[j];
			}
		}
		delete[] w;
	}
	else {
		int m = 0, n = S.rows - 1;

		s = V.rows * V.cols;
		for (i = 0; i < s; ++i) {
			V.data[i] = 0;
		}
		for (i = 0; i < V.rows; ++i) {
			V[i][i] = 1;
		}

		T* w = new T[A.rows];
		__LQ_HouseHolder(V, S, w);

		for (i = 0; i < S.rows; ++i) {
			for (j = 0; j <= i; ++j) {
				U[i][j] = S[i][j];
				S[i][j] = 0;
			}
			for (j = i + 1; j < S.rows; ++j) {
				U[i][j] = 0;
			}
		}

		bool flag = false;
		while (m < n) {
			flag = true;
			for (j = m + 1; j <= n; ++j) {
				flag = flag && __SVD_Jacobi_Rot(V, U, m, j, eps);
			}
			if (flag) {
				++m;
				continue;
			}

			flag = true;
			for (j = m; j < n; ++j) {
				flag = flag && __SVD_Jacobi_Rot(V, U, j, n, eps);
			}
			if (flag) {
				--n;
				continue;
			}

			for (i = m + 1; i < n; ++i) {
				for (j = i + 1; j <= n; ++j) {
					__SVD_Jacobi_Rot(V, U, i, j, eps);
				}
			}
		}

		for (i = 0; i < U.cols; ++i) {
			w[i] = 0;
		}
		for (i = 0; i < U.cols; ++i) {
			for (j = 0; j < U.cols; ++j) {
				w[j] += U[i][j] * U[i][j];
			}
		}
		for (i = 0; i < S.rows; ++i) {
			S[i][i] = sqrt(w[i]);
			w[i] = 1 / S[i][i];
		}
		for (i = 0; i < U.cols; ++i) {
			for (j = 0; j < U.rows; ++j) {
				U[i][j] = U[i][j] * w[j];
			}
		}
		delete[] w;
	}
}