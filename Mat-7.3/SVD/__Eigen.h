
template <typename T, typename t>
void __SVD_EVD_QR(MAT<t>& A, MAT<T>& U, MAT<T>& S, MAT<T>& V, double eps) {
	int i;
	if (A.rows > A.cols) {
		//_T(A)*A = V*S*_T(V)

		times_AT_B(A, A, S);

		__set0(V.data, V.rows * V.cols);
		for (i = 0; i < V.rows; ++i) {
			V[i][i] = 1;
		}

		T* w = U[0];
		T* v = U[1];
		__EVD_Q_HouseHolder(V, S, w, v);

		T* a(S.data);
		T* b(S.data + S.rows);
		for (i = 1; i < S.cols; ++i) {
			a[i] = S[i][i];
		}
		for (i = 1; i < S.cols - 1; ++i) {
			b[i] = S[i + 1][i];
		}
		__EVD_Q_Givens(a, b, V, 0, S.rows - 1, eps);

		for (i = 0; i < S.rows; ++i) {
			a[i] = sqrt(a[i]);
		}
		//U*S = A*V
		times_A_B(A, V, U);

		for (i = 0; i < U.cols; ++i) {
			__times(U[i], 1.0 / a[i], U[i], U.rows);
		}

		for (i = 1; i < S.rows; ++i) {
			S[i][i] = a[i];
		}
		for (i = 1; i < S.rows; ++i) {
			S[i - 1][i] = 0;
			S[i][i - 1] = 0;
		}
		for (i = 2; i < S.rows; ++i) {
			a[i] = 0;
		}
	}
	else {
		//A*_T(A) = U*S*_T(U)

		times_A_BT(A, A, S);

		__set0(U.data, U.rows * U.cols);
		for (i = 0; i < U.rows; ++i) {
			U[i][i] = 1;
		}
		T* w = V[0];
		T* v = V[1];
		__EVD_Q_HouseHolder(U, S, w, v);

		T* a(S.data);
		T* b(S.data + S.rows);
		for (i = 1; i < S.cols; ++i) {
			a[i] = S[i][i];
		}
		for (i = 1; i < S.cols - 1; ++i) {
			b[i] = S[i + 1][i];
		}
		__EVD_Q_Givens(a, b, U, 0, S.rows - 1, eps);

		for (i = 0; i < S.rows; ++i) {
			a[i] = sqrt(a[i]);
		}

		//V*S = _T(A)*U
		times_AT_B(A, U, V);

		for (i = 0; i < V.cols; ++i) {
			__times(V[i], 1.0 / a[i], V[i], V.rows);
		}

		for (i = 1; i < S.rows; ++i) {
			S[i][i] = a[i];
		}
		for (i = 1; i < S.rows; ++i) {
			S[i - 1][i] = 0;
			S[i][i - 1] = 0;
		}
		for (i = 2; i < S.rows; ++i) {
			a[i] = 0;
		}
	}
}

template <typename T, typename t>
void __SVD_EVD_Jacobi(MAT<t>& A, MAT<T>& U, MAT<T>& S, MAT<T>& V, double eps) {
	int i;
	if (A.rows > A.cols) {
		//_T(A)*A = V*S*_T(V)
		times_AT_B(A, A, S);

		__EVD_Q_Jacobi(S, V, eps);

		for (i = 0; i < A.cols; ++i) {
			S[i][i] = sqrt(S[i][i]);
		}

		//U*S = A*V
		times_A_B(A, V, U);
		for (i = 0; i < U.cols; ++i) {
			__times(U[i], 1.0 / S[i][i], U[i], U.rows);
		}
	}
	else {
		//A*_T(A) = U*S*_T(U)
		times_A_BT(A, A, S);

		__EVD_Q_Jacobi(S, U, eps);

		for (i = 0; i < A.rows; ++i) {
			S[i][i] = sqrt(S[i][i]);
		}

		//V*S = _T(A)*U
		times_AT_B(A, U, V);
		for (i = 0; i < V.cols; ++i) {
			__times(V[i], 1.0 / S[i][i], V[i], V.rows);
		}
	}
}
