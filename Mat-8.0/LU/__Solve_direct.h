
#define __BlockSize_solve 128

template<typename T>
void __Solve_direct_UX(__BlockMat<T>& U, __BlockMat<T>& X) {
	T* x;
	int i;
	for (int r = 0; r < X.cols; ++r) {
		x = X[r];
		for (i = U.cols - 1; i >= 0; --i) {
			x[i] /= U(i, i);
			__mixed(x, U[i], -x[i], x, i);
		}
	}
}

template <typename T>
void __Solve_direc_PLU(__BlockMat<T>& L, __BlockMat<T>& U, MAT<T>& X, int cL, int cU, int x) {
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
			X.swap_rows(r + x, j + x);
		}

		__times(Temp + r + 1, (T)1 / Temp[r], L[r] + r + 1, L.rows - r - 1);
	}
}

template <typename T>
void __Solve_direct(MAT<T>& A, MAT<T>& X) {
	int r, i, j, m, n;
	__BlockMat<T> _L, _U, _A;
	r = A.rows % __BlockSize_solve;

	if (r != 0) {
		_L.set(A, 0, 0, A.rows, r);
		_U.set(A, 0, 0, A.rows, r);
		__Solve_direc_PLU(_L, _U, X, 0, A.cols, 0);

		if (r < A.rows) {
			for (j = r; j < A.cols; j = j + __BlockSize_solve) {
				if (j + __BlockSize_solve > A.cols) n = A.cols - j;
				else n = __BlockSize_solve;

				_L.set(A, 0, 0, A.rows, r);
				_U.set(A, 0, j, r, n);
				__Solve_direct_LX(_L, _U);

				for (i = r; i < A.rows; i = i + __BlockSize_solve) {
					if (i + __BlockSize_solve > A.rows) m = A.rows - i;
					else m = __BlockSize_solve;

					_L.set(A, i, 0, m, r);
					_A.set(A, i, j, m, n);
					__times_A_B_LU(_L, _U, _A);
				}
			}
		}
	}
	for (; r < A.cols; r = r + __BlockSize_solve) {
		_L.set(A, r, r, A.rows - r, __BlockSize_solve);
		_U.set(A, r, r, A.rows - r, __BlockSize_solve);
		__Solve_direc_PLU(_L, _U, X, -r, A.cols - r, r);

		if (r + __BlockSize_solve < A.rows) {
			for (j = r + __BlockSize_solve; j < A.cols; j = j + __BlockSize_solve) {
				if (j + __BlockSize_solve > A.cols) n = A.cols - j;
				else n = __BlockSize_solve;

				_L.set(A, r, r, A.rows - r, __BlockSize_solve);
				_U.set(A, r, j, __BlockSize_solve, n);
				__Solve_direct_LX(_L, _U);

				for (i = r + __BlockSize_solve; i < A.rows; i = i + __BlockSize_solve) {
					if (i + __BlockSize_solve > A.rows) m = A.rows - i;
					else m = __BlockSize_solve;

					_L.set(A, i, r, m, __BlockSize_solve);
					_A.set(A, i, j, m, n);
					__times_A_B_LU(_L, _U, _A);
				}
			}
		}
	}

	r = A.rows % __BlockSize_solve;
	if (r != 0) {
		if (r < A.rows) {
			for (j = 0; j < X.cols; j = j + __BlockSize_solve) {
				if (j + __BlockSize_solve > X.cols) n = X.cols - j;
				else n = __BlockSize_solve;

				_L.set(A, 0, 0, A.rows, r);
				_U.set(X, 0, j, r, n);
				__Solve_direct_LX(_L, _U);

				for (i = r; i < X.rows; i = i + __BlockSize_solve) {
					if (i + __BlockSize_solve > X.rows) m = X.rows - i;
					else m = __BlockSize_solve;

					_L.set(A, i, 0, m, r);
					_A.set(X, i, j, m, n);
					__times_A_B_LU(_L, _U, _A);
				}
			}
		}
	}
	for (; r < A.cols; r = r + __BlockSize_solve) {
		for (j = 0; j < X.cols; j = j + __BlockSize_solve) {
			if (j + __BlockSize_solve > X.cols) n = X.cols - j;
			else n = __BlockSize_solve;

			_L.set(A, r, r, A.rows - r, __BlockSize_solve);
			_U.set(X, r, j, __BlockSize_solve, n);
			__Solve_direct_LX(_L, _U);

			for (i = r + __BlockSize_solve; i < X.rows; i = i + __BlockSize_solve) {
				if (i + __BlockSize_solve > X.rows) m = X.rows - i;
				else m = __BlockSize_solve;

				_L.set(A, i, r, m, __BlockSize_solve);
				_A.set(X, i, j, m, n);
				__times_A_B_LU(_L, _U, _A);
			}
		}
	}

	int k;
	r = (X.rows - 1) / __BlockSize_solve;
	for (j = r * __BlockSize_solve; j >= 0; j = j - __BlockSize_solve) {
		if (j + __BlockSize_solve < X.rows) n = __BlockSize_solve;
		else n = X.cols - j;

		for (k = 0; k < X.cols; k = k + __BlockSize_solve) {
			if (k + __BlockSize_solve > X.cols) m = X.cols - k;
			else m = __BlockSize_solve;

			_U.set(A, j, j, n, n);
			_L.set(X, j, k, n, m);
			__Solve_direct_UX(_U, _L);

			for (i = 0; i < j; i = i + __BlockSize_solve) {
				_U.set(A, i, j, __BlockSize_solve, n);
				_A.set(X, i, k, __BlockSize_solve, m);
				__times_A_B_LU(_U, _L, _A);
			}
		}
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

template<typename T>
T Det(MAT<T> n) {
	return __Det(n);
}

//X*A=B
template<typename T>
MAT<T> Solve_direct(MAT<T> A, MAT<T> B) {
	__Solve_direct(A, B);
	return B;
}

template <typename T>
void __Inv(MAT<T>& A, MAT<T>& X, int* pos) {
	int r, i, j, m, n;
	__BlockMat<T> _L, _U, _A;

	for (i = 0; i < A.rows; ++i) {
		pos[i] = i;
	}

	r = A.rows % __BlockSize_solve;

	if (r != 0) {
		_L.set(A, 0, 0, A.rows, r);
		_U.set(A, 0, 0, A.rows, r);
		__PLU2(pos, _L, _U, 0, A.cols);

		if (r < A.rows) {
			for (j = r; j < A.cols; j = j + __BlockSize_solve) {
				if (j + __BlockSize_solve > A.cols) n = A.cols - j;
				else n = __BlockSize_solve;

				_L.set(A, 0, 0, A.rows, r);
				_U.set(A, 0, j, r, n);
				__Solve_direct_LX(_L, _U);

				for (i = r; i < A.rows; i = i + __BlockSize_solve) {
					if (i + __BlockSize_solve > A.rows) m = A.rows - i;
					else m = __BlockSize_solve;

					_L.set(A, i, 0, m, r);
					_A.set(A, i, j, m, n);
					__times_A_B_LU(_L, _U, _A);
				}
			}
		}
	}
	for (; r < A.cols; r = r + __BlockSize_solve) {
		_L.set(A, r, r, A.rows - r, __BlockSize_solve);
		_U.set(A, r, r, A.rows - r, __BlockSize_solve);
		__PLU2(pos + r, _L, _U, -r, A.cols - r);

		if (r + __BlockSize_solve < A.rows) {
			for (j = r + __BlockSize_solve; j < A.cols; j = j + __BlockSize_solve) {
				if (j + __BlockSize_solve > A.cols) n = A.cols - j;
				else n = __BlockSize_solve;

				_L.set(A, r, r, A.rows - r, __BlockSize_solve);
				_U.set(A, r, j, __BlockSize_solve, n);
				__Solve_direct_LX(_L, _U);

				for (i = r + __BlockSize_solve; i < A.rows; i = i + __BlockSize_solve) {
					if (i + __BlockSize_solve > A.rows) m = A.rows - i;
					else m = __BlockSize_solve;

					_L.set(A, i, r, m, __BlockSize_solve);
					_A.set(A, i, j, m, n);
					__times_A_B_LU(_L, _U, _A);
				}
			}
		}
	}

	__set0(X.data, X.rows * X.cols);
	for (i = 0; i < X.rows; ++i) {
		X(i, pos[i]) = 1;
	}

	r = A.rows % __BlockSize_solve;
	if (r != 0) {
		if (r < A.rows) {
			for (j = 0; j < X.cols; j = j + __BlockSize_solve) {
				if (j + __BlockSize_solve > X.cols) n = X.cols - j;
				else n = __BlockSize_solve;

				_L.set(A, 0, 0, A.rows, r);
				_U.set(X, 0, j, r, n);
				__Solve_direct_LX(_L, _U);

				for (i = r; i < X.rows; i = i + __BlockSize_solve) {
					if (i + __BlockSize_solve > X.rows) m = X.rows - i;
					else m = __BlockSize_solve;

					_L.set(A, i, 0, m, r);
					_A.set(X, i, j, m, n);
					__times_A_B_LU(_L, _U, _A);
				}
			}
		}
	}
	for (; r < A.cols; r = r + __BlockSize_solve) {
		for (j = 0; j < X.cols; j = j + __BlockSize_solve) {
			if (j + __BlockSize_solve > X.cols) n = X.cols - j;
			else n = __BlockSize_solve;

			_L.set(A, r, r, A.rows - r, __BlockSize_solve);
			_U.set(X, r, j, __BlockSize_solve, n);
			__Solve_direct_LX(_L, _U);

			for (i = r + __BlockSize_solve; i < X.rows; i = i + __BlockSize_solve) {
				if (i + __BlockSize_solve > X.rows) m = X.rows - i;
				else m = __BlockSize_solve;

				_L.set(A, i, r, m, __BlockSize_solve);
				_A.set(X, i, j, m, n);
				__times_A_B_LU(_L, _U, _A);
			}
		}
	}

	int k;
	r = (X.rows - 1) / __BlockSize_solve;
	for (j = r * __BlockSize_solve; j >= 0; j = j - __BlockSize_solve) {
		if (j + __BlockSize_solve < X.rows) n = __BlockSize_solve;
		else n = X.cols - j;

		for (k = 0; k < X.cols; k = k + __BlockSize_solve) {
			if (k + __BlockSize_solve > X.cols) m = X.cols - k;
			else m = __BlockSize_solve;

			_U.set(A, j, j, n, n);
			_L.set(X, j, k, n, m);
			__Solve_direct_UX(_U, _L);

			for (i = 0; i < j; i = i + __BlockSize_solve) {
				_U.set(A, i, j, __BlockSize_solve, n);
				_A.set(X, i, k, __BlockSize_solve, m);
				__times_A_B_LU(_U, _L, _A);
			}
		}
	}
}

template<typename T>
MAT<T> Inv(MAT<T> A) {
	MAT<T> B(A.rows, A.cols);
	int* pos =(int*) _alloca(sizeof(int) * A.rows);
	__Inv(A, B, pos);

	return B;
}
