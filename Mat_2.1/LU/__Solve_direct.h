
#define __BlockSize_solve 64

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
void __Solve_direct_LU_X(MAT<T>& A, MAT<T>& X) {
	int r, i, j, l, m, n;
	__BlockMat<T> _L, _U, _A;
	
	for (r = 0; r < A.cols; r = r + __BlockSize_solve) {
		if (r + __BlockSize_solve > A.cols) l = A.cols - r;
		else l = __BlockSize_solve;

		for (j = 0; j < X.cols; j = j + __BlockSize_solve) {
			if (j + __BlockSize_solve > X.cols) n = X.cols - j;
			else n = __BlockSize_solve;

			_L.set(A, r, r, A.rows - r, l);
			_U.set(X, r, j, l, n);
			__Solve_direct_LX(_L, _U);

			for (i = r + __BlockSize_solve; i < X.rows; i = i + __BlockSize_solve) {
				if (i + __BlockSize_solve > X.rows) m = X.rows - i;
				else m = __BlockSize_solve;

				_L.set(A, i, r, m, l);
				_A.set(X, i, j, m, n);
				__times_A_B_LU(_L, _U, _A);
			}
		}
	}

	r = (X.rows - 1) / __BlockSize_solve;
	for (r = r * __BlockSize_solve; r >= 0; r = r - __BlockSize_solve) {
		if (r + __BlockSize_solve < X.rows) n = __BlockSize_solve;
		else n = X.cols - r;

		for (j = 0; j < X.cols; j = j + __BlockSize_solve) {
			if (j + __BlockSize_solve > X.cols) m = X.cols - j;
			else m = __BlockSize_solve;

			_U.set(A, r, r, n, n);
			_L.set(X, r, j, n, m);
			__Solve_direct_UX(_U, _L);

			for (i = 0; i < r; i = i + __BlockSize_solve) {
				_U.set(A, i, r, __BlockSize_solve, n);
				_A.set(X, i, j, __BlockSize_solve, m);
				__times_A_B_LU(_U, _L, _A);
			}
		}
	}
}

//X*A=B
template<typename T>
MAT<T> Solve_direct(MAT<T> A, MAT<T>& B) {
	MAT<T> b(B.rows, B.cols);
	int i, j;

	//int* pos = (int*)_alloca(sizeof(int) * B.rows);
	int* pos = new int[B.rows];

	for (i = 0; i < A.rows; ++i) {
		pos[i] = i;
	}

	__PLU(pos, A, A);

	T* Temp1, * Temp2;
	for (j = 0; j < B.cols; ++j) {
		Temp1 = b[j];
		Temp2 = B[j];
		for (i = 0; i < B.rows; ++i) {
			Temp1[i] = Temp2[pos[i]];
		}
	}
	delete[] pos;

	__Solve_direct_LU_X(A, b);

	return b;
}
template<typename T>
MAT<T> Inv2(MAT<T> A) {
	MAT<T> B(A.rows, A.cols);
	int i;

	//int* pos = (int*)_alloca(sizeof(int) * B.rows);
	int* pos = new int[B.rows];
	for (i = 0; i < A.rows; ++i) {
		pos[i] = i;
	}

	__PLU(pos, A, A);

	__set0(B.data, B.rows * B.cols);
	for (i = 0; i < A.cols; ++i) {
		B(i, pos[i]) = 1;
	}
	delete[] pos;

	__Solve_direct_LU_X(A, B);

	return B;
}





template<typename T>
void __Solve_direct_XU(__BlockMat<T>& U, __BlockMat<T>& X) {
	int i;
	for (int r = 0; r < X.cols; ++r) {
		for (i = r + 1; i < U.cols; ++i) {
			__mixed(X[i], X[r], -U(r, i), X[i], X.rows);
		}
	}
}
template<typename T>
void __Solve_direct_XL(__BlockMat<T>& L, __BlockMat<T>& X) {
	int i;
	for (int r = X.cols - 1; r > 0; --r) {
		for (i = 0; i < r; ++i) {
			__mixed(X[i], X[r], -L(r, i), X[i], X.rows);
		}
	}
}
template <typename T>
void __Solve_direct_X_LU(MAT<T>& A, MAT<T>& X) {
	int r, i, j, l, m, n;
	__BlockMat<T> _L, _U, _A;

	for (r = 0; r < A.cols; r = r + __BlockSize_solve) {
		if (r + __BlockSize_solve > A.cols) l = A.cols - r;
		else l = __BlockSize_solve;

		_U.set(A, r, r, l, l);

		for (i = 0; i < X.rows; i = i + __BlockSize_solve) {
			if (i + __BlockSize_solve > X.rows) m = X.rows - i;
			else m = __BlockSize_solve;

			_A.set(X, i, r, m, l);
			__Solve_direct_XU(_U, _A);
		}

		for (j = r + __BlockSize_solve; j < X.cols; j = j + __BlockSize_solve) {
			if (j + __BlockSize_solve > X.cols) n = X.cols - j;
			else n = __BlockSize_solve;

			_U.set(A, r, j, l, n);

			for (i = 0; i < X.rows; i = i + __BlockSize_solve) {
				if (i + __BlockSize_solve > X.rows) m = X.rows - i;
				else m = __BlockSize_solve;

				_L.set(X, i, r, m, l);
				_A.set(X, i, j, m, n);
				__times_A_B_LU(_L, _U, _A);
			}
		}
	}

	r = (X.rows - 1) / __BlockSize_solve;
	for (r = r * __BlockSize_solve; r >= 0; r = r - __BlockSize_solve) {
		if (r + __BlockSize_solve < X.rows) n = __BlockSize_solve;
		else n = X.cols - r;

		_L.set(A, r, r, n, n);
		for (i = 0; i < A.rows; i = i + __BlockSize_solve) {
			if (i + __BlockSize_solve > X.rows) m = X.rows - i;
			else m = __BlockSize_solve;

			_A.set(X, i, r, m, n);
			__Solve_direct_XL(_L, _A);
		}

		for (j = 0; j < r; j = j + __BlockSize_solve) {

			_L.set(A, r, j, n, __BlockSize_solve);

			for (i = 0; i < A.rows; i = i + __BlockSize_solve) {
				if (i + __BlockSize_solve > X.rows) m = X.rows - i;
				else m = __BlockSize_solve;

				_U.set(X, i, r, m, n);
				_A.set(X, i, j, m, __BlockSize_solve);
				__times_A_B_LU(_U, _L, _A);
			}
		}
	}
}

template<typename T>
void __Inv_swap(MAT<T>& A, int i, int* pos, int j, bool* flag) {
	int k = pos[j];
	if (k != i) {
		__Inv_swap(A, i, pos, k, flag);
	}
	__copy(A[k], A[j], A.rows);
	flag[j] = false;
}

template<typename T>
MAT<T> Inv(MAT<T> A) {
	MAT<T> B(A.rows, A.cols);
	int i;

	//int* pos = (int*)_alloca(sizeof(int) * B.rows);
	int* pos = new int[B.rows];
	for (i = 0; i < A.rows; ++i) {
		pos[i] = i;
	}

	__PLU(pos, A, A);

	__set0(B.data, B.rows * B.cols);
	T temp;
	for (i = 0; i < A.cols; ++i) {
		temp = 1 / A(i, i);
		__times(A[i], temp, A[i], i);
		B(i, i) = temp;
	}

	__Solve_direct_X_LU(A, B);

	bool* if_nswap = new bool[B.cols];
	T* Temp = new T[B.rows];

	for (i = 0; i < B.cols; ++i) {
		if_nswap[i] = true;
	}
	int j;
	for (i = 0; i < B.cols; ++i) {
		if (if_nswap[i]) {
			j = pos[i];
			if (i != j) {
				__copy(Temp, B[i], B.rows);
				__Inv_swap(B, i, pos, i, if_nswap);
				__copy(B[j], Temp, B.rows);
			}
		}
	}
	delete[] pos;
	delete[] if_nswap;
	delete[] Temp;
	return B;
}