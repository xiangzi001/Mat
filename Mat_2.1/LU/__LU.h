
#define __BlockSize_LU 64

template<typename t>
void __times_A_B_LU(__BlockMat<t>& A, __BlockMat<t>& B, __BlockMat<t>& C) {
	int i, j;

	for (j = 0; j < C.cols; j = j + __micro_kernel_size<t, t, t>()) {
		if (j + __micro_kernel_size<t, t, t>() > C.cols) {
			int k;
			for (; j < C.cols; ++j) {
				t* b = B[j];
				for (k = 0; k < A.cols; ++k) {
					__mixed(C[j], A[k], -b[k], C[j], C.rows);
				}
			}
			break;
		}
		else {
			for (i = 0; i < C.rows; i = i + __micro_kernel_size<t, t, t>()) {
				if (i + __micro_kernel_size<t, t, t>() > C.rows) {
					int y, k;
					for (y = j; y < j + __micro_kernel_size<t, t, t>(); ++y) {
						t* b = B[y];
						for (k = 0; k < A.cols; ++k) {
							__mixed(C[y] + i, A[k] + i, -b[k], C[y] + i, C.rows - i);
						}
					}
				}
				else __micro_kernel_ntimes_A_B(A.data + i, A._rows, B[j], B._rows, C[j] + i, C._rows, A.cols);
			}
		}
	}
}

template<typename T>
void __Solve_direct_LX(__BlockMat<T>& L, __BlockMat<T>& X) {
	T* x;
	int i;
	for (int r = 0; r < X.cols; ++r) {
		x = X[r];
		for (i = 1; i < L.cols; ++i) {
			__mixed(x + i, L[i - 1] + i, -x[i - 1], x + i, X.rows - i);
		}
	}
}

template <typename T>
void __LU2(__BlockMat<T>& L, __BlockMat<T>& U) {
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

template <typename T>
void __LU(MAT<T>& L, MAT<T>& U) {
	int r, s, i, j, m, n;
	__BlockMat<T> _L, _U, _A;
	if (U.rows > U.cols) {
		s = U.cols;
	}
	else {
		s = U.rows;
	}
	r = s % __BlockSize_LU;

	if (r != 0) {
		_L.set(L, 0, 0, L.rows, r);
		_U.set(U, 0, 0, U.rows, r);
		__LU2(_L, _U);

		for (j = r; j < U.cols; j = j + __BlockSize_LU) {
			if (j + __BlockSize_LU > U.cols) n = U.cols - j;
			else n = __BlockSize_LU;

			_L.set(L, 0, 0, L.rows, r);
			_U.set(U, 0, j, r, n);
			__Solve_direct_LX(_L, _U);

			for (i = r; i < U.rows; i = i + __BlockSize_LU) {
				if (i + __BlockSize_LU > U.rows) m = U.rows - i;
				else m = __BlockSize_LU;

				_L.set(L, i, 0, m, r);
				_A.set(U, i, j, m, n);
				__times_A_B_LU(_L, _U, _A);
			}
		}
	}
	for (; r < s; r = r + __BlockSize_LU) {
		_L.set(L, r, r, L.rows - r, __BlockSize_LU);
		_U.set(U, r, r, U.rows - r, __BlockSize_LU);
		__LU2(_L, _U);

		if (r + __BlockSize_LU < s) {
			for (j = r + __BlockSize_LU; j < U.cols; j = j + __BlockSize_LU) {
				if (j + __BlockSize_LU > U.cols) n = U.cols - j;
				else n = __BlockSize_LU;

				_L.set(L, r, r, L.rows - r, __BlockSize_LU);
				_U.set(U, r, j, __BlockSize_LU, n);
				__Solve_direct_LX(_L, _U);

				for (i = r + __BlockSize_LU; i < U.rows; i = i + __BlockSize_LU) {
					if (i + __BlockSize_LU > U.rows) m = U.rows - i;
					else m = __BlockSize_LU;

					_L.set(L, i, r, m, __BlockSize_LU);
					_A.set(U, i, j, m, n);
					__times_A_B_LU(_L, _U, _A);
				}
			}
		}
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




template <typename T>
void __PLU2(int* pos, __BlockMat<T>& L, __BlockMat<T>& U, int cL, int cU) {
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
			L.swap_rows(r, j, cL, r);
			U.swap_rows(r, j, r, cU);
			i = pos[j];
			pos[j] = pos[r];
			pos[r] = i;
		}

		__times(Temp + r + 1, (T)1 / Temp[r], L[r] + r + 1, L.rows - r - 1);
	}
}

template <typename T>
void __PLU(int* pos, MAT<T>& L, MAT<T>& U) {
	int r, s, i, j, m, n;
	__BlockMat<T> _L, _U, _A;
	if (U.rows > U.cols) {
		s = U.cols;
	}
	else {
		s = U.rows;
	}
	r = s % __BlockSize_LU;

	if (r != 0) {
		_L.set(L, 0, 0, L.rows, r);
		_U.set(U, 0, 0, U.rows, r);
		__PLU2(pos, _L, _U, 0, U.cols);

		for (j = r; j < U.cols; j = j + __BlockSize_LU) {
			if (j + __BlockSize_LU > U.cols) n = U.cols - j;
			else n = __BlockSize_LU;

			_L.set(L, 0, 0, L.rows, r);
			_U.set(U, 0, j, r, n);
			__Solve_direct_LX(_L, _U);

			for (i = r; i < U.rows; i = i + __BlockSize_LU) {
				if (i + __BlockSize_LU > U.rows) m = U.rows - i;
				else m = __BlockSize_LU;

				_L.set(L, i, 0, m, r);
				_A.set(U, i, j, m, n);
				__times_A_B_LU(_L, _U, _A);
			}
		}
	}
	for (; r < s; r = r + __BlockSize_LU) {
		_L.set(L, r, r, L.rows - r, __BlockSize_LU);
		_U.set(U, r, r, U.rows - r, __BlockSize_LU);
		__PLU2(pos + r, _L, _U, -r, U.cols - r);

		if (r + __BlockSize_LU < s) {
			for (j = r + __BlockSize_LU; j < U.cols; j = j + __BlockSize_LU) {
				if (j + __BlockSize_LU > U.cols) n = U.cols - j;
				else n = __BlockSize_LU;

				_L.set(L, r, r, L.rows - r, __BlockSize_LU);
				_U.set(U, r, j, __BlockSize_LU, n);
				__Solve_direct_LX(_L, _U);

				for (i = r + __BlockSize_LU; i < U.rows; i = i + __BlockSize_LU) {
					if (i + __BlockSize_LU > U.rows) m = U.rows - i;
					else m = __BlockSize_LU;

					_L.set(L, i, r, m, __BlockSize_LU);
					_A.set(U, i, j, m, n);
					__times_A_B_LU(_L, _U, _A);
				}
			}
		}
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

template <typename t>
void PLU_getP(int* pos, MAT<t>& P) {
	__set0(P.data, P.rows * P.cols);
	for (int i = 0; i < P.rows; ++i) {
		P(pos[i], i) = 1;
	}
}

template<typename T>
T __Det(MAT<T>& A) {
	int x, y, i, pos;
	T big, now;
	T r = 1, temp;
	for (y = 0; y < A.rows; ++y) {
		pos = y;
		big = fabs(A[y][y]);
		for (i = y + 1; i < A.cols; ++i) {
			now = fabs(A[i][y]);
			if (now > big) {
				pos = i;
				big = now;
			}
		}
		if (big == 0) return 0;
		if (pos != y) {
			__swap(A[pos] + y, A[y] + y, A.rows - y);
			r = -r;
		}
		for (x = y + 1; x < A.cols; ++x) {
			temp = -A[x][y] / A[y][y];
			__mixed(A[x] + y, A[y] + y, temp, A[x] + y, A.rows - y);
		}
	}
	for (x = 0; x < A.cols; ++x)
		r = r * A[x][x];

	return r;
}
//*/



template <typename T>
void __Det(T& ret, __BlockMat<T>& A, int cL, int cU) {
	int r, i;
	T* Temp;
	for (r = 0; r < A.cols; ++r) {
		Temp = A[r];
		for (i = 0; i < r; ++i) {
			__mixed(Temp + i + 1, A[i] + i + 1, -Temp[i], Temp + i + 1, A.rows - i - 1);
		}

		int j = r;
		auto temp1 = fabs(A(r, r)), temp2 = temp1;
		for (i = r + 1; i < A.rows; ++i) {
			temp2 = fabs(A(i, r));
			if (temp2 > temp1) {
				temp1 = temp2;
				j = i;
			}
		}
		if (j != r) {
			A.swap_rows(r, j, cL, cU);
			ret = -ret;
		}

		__times(Temp + r + 1, (T)1 / Temp[r], A[r] + r + 1, A.rows - r - 1);
	}
}

template<typename T>
T Det(MAT<T> A) {
	T ret = 1;

	int r, i, j, m, n;
	__BlockMat<T> _L, _U, _A;

	r = A.cols % __BlockSize_LU;

	if (r != 0) {
		_A.set(A, 0, 0, A.rows, r);
		__Det(ret, _A, 0, A.cols);

		for (j = r; j < A.cols; j = j + __BlockSize_LU) {
			if (j + __BlockSize_LU > A.cols) n = A.cols - j;
			else n = __BlockSize_LU;

			_L.set(A, 0, 0, A.rows, r);
			_U.set(A, 0, j, r, n);
			__Solve_direct_LX(_L, _U);

			for (i = r; i < A.rows; i = i + __BlockSize_LU) {
				if (i + __BlockSize_LU > A.rows) m = A.rows - i;
				else m = __BlockSize_LU;

				_L.set(A, i, 0, m, r);
				_A.set(A, i, j, m, n);
				__times_A_B_LU(_L, _U, _A);
			}
		}
	}
	for (; r < A.cols; r = r + __BlockSize_LU) {
		_A.set(A, r, r, A.rows - r, __BlockSize_LU);
		__Det(ret, _A, -r, A.cols - r);

		if (r + __BlockSize_LU < A.cols) {
			for (j = r + __BlockSize_LU; j < A.cols; j = j + __BlockSize_LU) {
				if (j + __BlockSize_LU > A.cols) n = A.cols - j;
				else n = __BlockSize_LU;

				_L.set(A, r, r, A.rows - r, __BlockSize_LU);
				_U.set(A, r, j, __BlockSize_LU, n);
				__Solve_direct_LX(_L, _U);

				for (i = r + __BlockSize_LU; i < A.rows; i = i + __BlockSize_LU) {
					if (i + __BlockSize_LU > A.rows) m = A.rows - i;
					else m = __BlockSize_LU;

					_L.set(A, i, r, m, __BlockSize_LU);
					_A.set(A, i, j, m, n);
					__times_A_B_LU(_L, _U, _A);
				}
			}
		}
	}
	for (r = 0; r < A.cols; ++r) {
		ret = ret * A(r, r);
	}
	return ret;
}
