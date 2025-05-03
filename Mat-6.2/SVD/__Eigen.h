
template <typename T>
void __SVD_EVD_QR_1(MAT<T>& A, MAT<T>& U, MAT<T>& S, MAT<T>& V, double eps) {
	int i, j;
	if (A.rows > A.cols) {
		T* w = new T[A.cols + 64 / sizeof(T) - 1];
		//_T(A)*A = V*S*_T(V)

		times_AT_B(A, A, S);

		__EVD_Q_QR_test_1(S, V, S.data, w, eps);

		for (i = 0; i < S.rows; ++i) {
			S[i][i] = sqrt(S.data[i]);
		}
		for (i = 1; i < S.rows; ++i) {
			S[i - 1][i] = 0;
			S[i][i - 1] = 0;
		}
		for (i = 2; i < S.rows; ++i) {
			S.data[i] = 0;
		}

		for (i = 0; i < S.rows; ++i) {
			S[i][i] = sqrt(w[i]);
		}
		//U*S = A*V
		times_A_B(A, V, U, w);

		for (i = 0; i < S.rows; ++i) {
			w[i] = S[i][i];
		}
		__SVD_Q_times(U, w, w);
		delete[] w;
	}
	else {
		T* w = new T[A.rows + 64 / sizeof(T) - 1];
		//A*_T(A) = U*S*_T(U)

		times_A_BT(A, A, S);

		__EVD_Q_QR_test_1(S, U, S.data, w, eps);

		for (i = 0; i < S.rows; ++i) {
			w[i] = sqrt(S.data[i]);
			S[i][i] = w[i];
		}
		for (i = 1; i < S.rows; ++i) {
			S[i - 1][i] = 0;
			S[i][i - 1] = 0;
		}
		for (i = 2; i < S.rows; ++i) {
			S.data[i] = 0;
		}

		//V*S = _T(A)*U
		times_AT_B(A, U, V);
		__SVD_Q_times(V, w, w);

		delete[] w;
	}
}

template <typename T>
void __SVD_EVD_QR_2(MAT<T>& A, MAT<T>& U, MAT<T>& S, MAT<T>& V, double eps) {
	int i, j;
	if (A.rows > A.cols) {
		T* w = new T[A.cols + 64 / sizeof(T) - 1];
		//_T(A)*A = V*S*_T(V)

		times_AT_B(A, A, S);

		__EVD_Q_QR_test_2(S, V, w, eps);

		for (i = 0; i < S.rows; ++i) {
			S[i][i] = sqrt(w[i]);
		}
		//U*S = A*V
		times_A_B(A, V, U, w);

		for (i = 0; i < S.rows; ++i) {
			w[i] = S[i][i];
		}
		__SVD_Q_times(U, w, w);
		delete[] w;
	}
	else {
		T* w = new T[A.rows + 64 / sizeof(T) - 1];
		//A*_T(A) = U*S*_T(U)

		times_A_BT(A, A, S);

		__EVD_Q_QR_test_2(S, U, w, eps);

		for (i = 0; i < S.rows; ++i) {
			w[i] = sqrt(w[i]);
			S[i][i] = w[i];
		}

		//V*S = _T(A)*U
		times_AT_B(A, U, V);
		__SVD_Q_times(V, w, w);

		delete[] w;
	}
}

template <typename T>
void __SVD_EVD_Jacobi(MAT<T>& A, MAT<T>& U, MAT<T>& S, MAT<T>& V, double eps) {
	if (A.rows > A.cols) {
		//_T(A)*A = V*S*_T(V)
		times_AT_B(A, A, S);

		__EVD_Q_Jacobi(S, V, eps);

		for (int i = 0; i < A.cols; ++i) {
			S[i][i] = sqrt(S[i][i]);
		}

		//U*S = A*V
		times_A_B(A, V, U);
		for (int i = 0; i < A.rows; ++i) {
			for (int j = 0; j < A.cols; ++j) {
				U[i][j] = U[i][j] / S[j][j];
			}
		}
	}
	else {
		//A*_T(A) = U*S*_T(U)
		MAT<T> B(A.cols, A.rows);
		B.rows = A.rows;

		times_A_BT(A, A, B);

		__EVD_Q_Jacobi(B, U, eps);

		for (int i = 0; i < A.rows; ++i) {
			S[i][i] = sqrt(B[i][i]);
		}

		//V*S = _T(A)*U
		B.rows = A.cols;
		times_AT_B(A, U, B);
		for (int i = 0; i < A.cols; ++i) {
			for (int j = 0; j < A.rows; ++j) {
				V[i][j] = B[i][j] / S[j][j];
			}
		}
	}
}
