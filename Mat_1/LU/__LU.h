
template <typename T>
void __LU(MAT<T>& L, MAT<T>& U) {
	int r, i;
	T* Temp;
	for (r = 0; r < U.cols; ++r) {
		Temp = U[r];
		for (i = 0; i < r; ++i) {
			__mixed(Temp + i + 1, L[i] + i + 1, -Temp[i], Temp + i + 1, U.rows - i - 1);
		}
		__times(Temp + r + 1, (T)1 / Temp[r], L[r] + r + 1, L.rows - r - 1);
	}
}

template <typename t1, typename t2>
void LU(MAT<t1>& A, MAT<t2>& L, MAT<t2>& U) {
	if (A.data != U.data)
		__copy(U.data, A.data, U.rows * U.cols);
	__LU(L, U);
	if (L.data != U.data) {
		for (int i = 0; i < L.rows; ++i) {
			__set0(L[i], i);
		}
		for (int i = 0; i < L.rows; ++i) {
			L[i][i] = 1;
		}
		for (int i = 0; i < L.rows; ++i) {
			__set0(U[i] + i + 1, U.rows - i - 1);
		}
	}
}


template <typename T>
void __PLU(int* pos, MAT<T>& L, MAT<T>& U) {
	int r, i;
	T* Temp;
	for (r = 0; r < U.cols; ++r) {
		Temp = U[r];
		for (i = 0; i < r; ++i) {
			__mixed(Temp + i + 1, L[i] + i + 1, -Temp[i], Temp + i + 1, U.rows - i - 1);
		}

		int j = r;
		auto temp1 = fabs(U(r, r)), temp2 = temp1;
		for (i = r + 1; i < U.rows; ++i) {
			temp2 = fabs(U(i, r));
			if (temp2 > temp1) {
				temp1 = temp2;
				j = i;
			}
		}
		if (j != r) {
			L.swap_rows(r, j, 0, r);
			U.swap_rows(r, j, r, U.cols);
			i = pos[j];
			pos[j] = pos[r];
			pos[r] = i;
		}

		if(Temp[r] != 0)
			__times(Temp + r + 1, (T)1 / Temp[r], L[r] + r + 1, L.rows - r - 1);
	}
}

template <typename t1, typename t2>
void PLU(MAT<t1>& A, int* pos, MAT<t2>& L, MAT<t2>& U) {
	if (A.data != U.data)
		__copy(U.data, A.data, U.rows * U.cols);
	for (int i = 0; i < U.rows; ++i) {
		pos[i] = i;
	}

	__PLU(pos, L, U);
	if (L.data != U.data) {
		for (int i = 0; i < L.rows; ++i) {
			__set0(L[i], i);
		}
		for (int i = 0; i < L.rows; ++i) {
			L[i][i] = 1;
		}
		for (int i = 0; i < L.rows; ++i) {
			__set0(U[i] + i + 1, U.rows - i - 1);
		}
	}
}

template <typename t1, typename t2>
void LU_getLU(MAT<t1>& LU, MAT<t2>& L, MAT<t2>& U) {
	int i;
	int s = min(LU.rows, LU.cols);
	for (i = 0; i < s; ++i) {
		__set0(L[i], i);
		L(i, i) = 1;
		__copy(L[i] + i + 1, LU[i] + i + 1, L.rows - i - 1);
	}
	for (i = s; i < L.cols; ++i) {
		__set0(L[i], L.rows);
		L(i, i) = 1;
	}

	for (i = 0; i < s; ++i) {
		__copy(U[i], LU[i], i + 1);
		__set0(U[i] + i + 1, U.rows - i - 1);
	}
	for (i = s; i < U.cols; ++i) {
		__copy(U[i], LU[i], U.rows);
	}
}

template <typename t>
void PLU_getP(int* pos, MAT<t>& P) {
	__set0(P.data, P.rows * P.cols);
	for (int i = 0; i < P.rows; ++i) {
		P(pos[i], i) = 1;
	}
}
