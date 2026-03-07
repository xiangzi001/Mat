
template <typename T>
void __LU1(MAT<T>& L, MAT<T>& U) {
	int r, i;
	for (r = 0; r < U.cols; ++r) {
		__times(U[r] + r + 1, 1.0 / U(r, r), L[r] + r + 1, L.rows - r - 1);
		for (i = r; i < U.cols; ++i) {
			__mixed2(U[i] + r + 1, L[r] + r + 1, -U(r, i), U[i] + r + 1, U.rows - r - 1);
		}
	}
}
template <typename T>
void __LU2(MAT<T>& LU) {
	int r, i;
	for (r = 0; r < LU.cols; ++r) {
		__times(LU[r] + r + 1, 1.0 / LU(r, r), LU[r] + r + 1, LU.rows - r - 1);
		for (i = r + 1; i < LU.cols; ++i) {
			__mixed2(LU[i] + r + 1, LU[r] + r + 1, -LU(r, i), LU[i] + r + 1, LU.rows - r - 1);
		}
	}
}

template <typename t1, typename t2>
void LU(MAT<t1>& A, MAT<t2>& L, MAT<t2>& U) {
	if (A.data != U.data)
	__copy(U.data, A.data, U.rows * U.cols);
	if (L.data != U.data) {
		__set0(L.data, L.rows * L.cols);
		for (int i = 0; i < L.rows; ++i) {
			L[i][i] = 1;
		}
		__LU1(L, U);
	}
	else {
		__LU2(U);
	}
}
