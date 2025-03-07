
template <typename T>
bool __SVD_Jacobi_U(MAT<T>& A, MAT<T>& U, int& i, int& j, T& eps) {
	int k = 0;
	T b = 0;
	T a = 0;

	for (k = 0; k < A.cols; ++k) {
		b = b - A[i][k] * A[i][k];
	}
	for (k = 0; k < A.cols; ++k) {
		a = a + A[i][k] * A[j][k];
	}
	for (k = 0; k < A.cols; ++k) {
		b = b + A[j][k] * A[j][k];
	}

	//T t = (-b + sqrt(b * b + 4 * a * a)) / (2 * a);
	//T c = 1 / sqrt(t * t + 1);
	//T s = t * c;

	double Angle = 0.5 * atan2(2 * a, b);
	T c = cos(Angle);
	T s = sin(Angle);

	for (k = 0; k < A.cols; ++k) {
		a = A[i][k] * c - A[j][k] * s;
		b = A[i][k] * s + A[j][k] * c;
		A[i][k] = a;
		A[j][k] = b;
	}
	for (k = 0; k < U.rows; ++k) {
		a = U[k][i] * c - U[k][j] * s;
		b = U[k][i] * s + U[k][j] * c;
		U[k][i] = a;
		U[k][j] = b;
	}

	if (s < eps) {
		return false;
	}
	else {
		return true;
	}
}

template <typename T>
bool __SVD_Jacobi_V(MAT<T>& A, MAT<T>& V, int& i, int& j, T& eps) {
	int k = 0;
	T b = 0;
	T a = 0;

	for (k = 0; k < A.rows; ++k) {
		b = b - A[k][i] * A[k][i];
		a = a + A[k][i] * A[k][j];
		b = b + A[k][j] * A[k][j];
	}

	//T t = (-b + sqrt(b * b + 4 * a * a)) / (2 * a);
	//T c = 1 / sqrt(t * t + 1);
	//T s = t * c;

	double Angle = 0.5 * atan2(2 * a, b);
	T c = cos(Angle);
	T s = sin(Angle);

	for (k = 0; k < A.rows; ++k) {
		a = A[k][i] * c - A[k][j] * s;
		b = A[k][i] * s + A[k][j] * c;
		A[k][i] = a;
		A[k][j] = b;
	}
	for (k = 0; k < V.rows; ++k) {
		a = V[k][i] * c - V[k][j] * s;
		b = V[k][i] * s + V[k][j] * c;
		V[k][i] = a;
		V[k][j] = b;
	}

	if (s < eps) {
		return false;
	}
	else {
		return true;
	}
}

template <typename T>
void __SVD_Jacobi(MAT<T>& A, MAT<T>& U, MAT<T>& S, MAT<T>& V, double eps) {
	int i;
	int j;
	int s = S.rows * S.cols;
	for (i = 0; i < s; ++i) {
		S.data[i] = A.data[i];
	}

	if (A.rows > A.cols) {
		s = U.rows * U.cols;
		for (i = 0; i < s; ++i) {
			U.data[i] = 0;
		}
		for (i = 0; i < U.rows; ++i) {
			U[i][i] = 1;
		}

		bool flag = true;
		while (flag) {
			flag = false;
			for (i = 0; i < S.cols; ++i) {
				for (j = i + 1; j < S.rows; ++j) {
					if (__SVD_Jacobi_U(S, U, j, i, eps)) {
						flag = true;
					}
				}
			}
		}

		T temp;
		for (i = 0; i < S.cols; ++i) {
			temp = 0;
			for (j = 0; j < S.cols; ++j) {
				temp += S[i][j] * S[i][j];
			}
			temp = 1.0 / sqrt(temp);

			for (j = 0; j < V.rows; ++j) {
				V[j][i] = S[i][j] * temp;
			}
			for (j = 0; j < S.cols; ++j) {
				S[i][j] = 0;
			}
			S[i][i] = 1.0 / temp;
		}
	}
	else {
		s = V.rows * V.cols;
		for (i = 0; i < s; ++i) {
			V.data[i] = 0;
		}
		for (i = 0; i < V.rows; ++i) {
			V[i][i] = 1;
		}

		bool flag = true;
		while (flag) {
			flag = false;
			for (i = 0; i < S.rows; ++i) {
				for (j = i + 1; j < S.cols; ++j) {
					if (__SVD_Jacobi_V(S, V, j, i, eps)) {
						flag = true;
					}
				}
			}
		}

		T temp;
		for (j = 0; j < S.rows; ++j) {
			temp = 0;
			for (i = 0; i < S.rows; ++i) {
				temp += S[i][j] * S[i][j];
			}
			temp = 1.0 / sqrt(temp);

			for (i = 0; i < U.rows; ++i) {
				U[i][j] = S[i][j] * temp;
			}
			for (i = 0; i < S.rows; ++i) {
				S[i][j] = 0;
			}
			S[j][j] = 1.0 / temp;
		}
	}
}