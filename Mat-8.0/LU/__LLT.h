
#define __BlockSize_LLT 128

#ifdef XiangZiMatImport_SIMD

void __times_A_BT_LLT(__BlockMat<double>& A, __BlockMat<double>& B, __BlockMat<double>& C) {
	int i, j;

	for (j = 0; j < C.cols; j = j + 8) {
		if (j + 8 > C.cols) {
			int k;
			for (; j < C.cols; ++j) {
				for (k = 0; k < A.cols; ++k) {
					__mixed(C[j], A[k], -B(j, k), C[j], C.rows);
				}
			}
			break;
		}
		else {
			for (i = 0; i < C.rows; i = i + 8) {
				if (i + 8 > C.rows) {
					int y, k;
					for (y = j; y < j + 8; ++y) {
						for (k = 0; k < A.cols; ++k) {
							__mixed(C[y] + i, A[k] + i, -B(y, k), C[y] + i, C.rows - i);
						}
					}
				}
				else __micro_kernel_8x8_ntimes_A_BT(A.data + i, A._rows, B.data + j, B._rows, C[j] + i, C._rows, A.cols);
			}
		}
	}
}
void __times_A_BT_LLT2(__BlockMat<double>& A, __BlockMat<double>& C) {
	int i, j;

	for (j = 0; j < C.cols; j = j + 8) {
		for (i = j; i < C.rows; i = i + 8) {
			if (i + 8 > C.rows) {
				int y, k, n;
				if (j + 8 > C.cols) n = C.cols;
				else n = j + 8;

				for (y = j; y < n; ++y) {
					for (k = 0; k < A.cols; ++k) {
						__mixed(C[y] + i, A[k] + i, -A(y, k), C[y] + i, C.rows - i);
					}
				}
			}
			else __micro_kernel_8x8_ntimes_A_BT(A.data + i, A._rows, A.data + j, A._rows, C[j] + i, C._rows, A.cols);
		}
	}
}
#endif

template<typename t>
void __times_A_BT_LLT(__BlockMat<t>& A, __BlockMat<t>& B, __BlockMat<t>& C) {
	int i, j;

	for (j = 0; j < C.cols; j = j + 4) {
		if (j + 4 > C.cols) {
			int k;
			for (; j < C.cols; ++j) {
				for (k = 0; k < A.cols; ++k) {
					__mixed(C[j], A[k], -B(j, k), C[j], C.rows);
				}
			}
			break;
		}
		else {
			for (i = 0; i < C.rows; i = i + 4) {
				if (i + 4 > C.rows) {
					int y, k;
					for (y = j; y < j + 4; ++y) {
						for (k = 0; k < A.cols; ++k) {
							__mixed(C[y] + i, A[k] + i, -B(y, k), C[y] + i, C.rows - i);
						}
					}
				}
				else __micro_kernel_4x4_ntimes_A_BT(A.data + i, A._rows, B.data + j, B._rows, C[j] + i, C._rows, A.cols);
			}
		}
	}
}
template<typename t>
void __times_A_BT_LLT2(__BlockMat<t>& A, __BlockMat<t>& C) {
	int i, j;

	for (j = 0; j < C.cols; j = j + 4) {
		for (i = j; i < C.rows; i = i + 4) {
			if (i + 4 > C.rows) {
				int y, k, n;
				if (j + 4 > C.cols) n = C.cols;
				else n = j + 4;
				
				for (y = j; y < n; ++y) {
					for (k = 0; k < A.cols; ++k) {
						__mixed(C[y] + i, A[k] + i, -A(y, k), C[y] + i, C.rows - i);
					}
				}
			}
			else __micro_kernel_4x4_ntimes_A_BT(A.data + i, A._rows, A.data + j, A._rows, C[j] + i, C._rows, A.cols);
		}
	}
}
template<typename T>
void __Solve_direct_XLT(__BlockMat<T>& L, __BlockMat<T>& X) {
	T* x;
	int j, k;
	for (j = 0; j < X.cols; ++j) {
		x = X[j];
		__times(x, (T)1 / L(j, j), x, X.rows);
		for (k = j + 1; k < X.cols; ++k) {
			__mixed(X[k], x, -L(k, j), X[k], X.rows);
		}
	}
}
template <typename t>
void __LLT2(__BlockMat<t>& L) {
	t temp;
	int i, j;
	for (j = 0; j < L.cols; ++j) {
		temp = (t)1 / sqrt(L(j, j));
		__times(L[j] + j, temp, L[j] + j, L.rows - j);
		for (i = j + 1; i < L.rows; ++i) {
			__mixed(L[i] + i, L[j] + i, -L(i, j), L[i] + i, L.rows - i);
		}
	}
}
template <typename t>
void __LLT(MAT<t>& L) {
	int i, j, k;
	int m, n, l;
	__BlockMat<t> _L, _L1, _L2;

	for (j = 0; j < L.cols; j = j + __BlockSize_LLT) {
		if (j + __BlockSize_LLT > L.cols) n = L.cols - j;
		else n = __BlockSize_LLT;

		_L.set(L, j, j, n, n);
		__LLT2(_L);

		for (i = j + __BlockSize_LLT; i < L.cols; i = i + __BlockSize_LLT) {
			if (i + __BlockSize_LLT > L.cols) m = L.cols - i;
			else m = __BlockSize_LLT;
			_L1.set(L, i, j, m, n);
			__Solve_direct_XLT(_L, _L1);
		}

		for (i = j + __BlockSize_LLT; i < L.cols; i = i + __BlockSize_LLT) {
			if (i + __BlockSize_LLT > L.cols) m = L.cols - i;
			else m = __BlockSize_LLT;

			//k = i
			_L1.set(L, i, j, m, n);
			_L.set(L, i, i, m, m);
			__times_A_BT_LLT2(_L1, _L);

			for (k = i + __BlockSize_LLT; k < L.cols; k = k + __BlockSize_LLT) {
				if (k + __BlockSize_LLT > L.cols) l = L.cols - k;
				else l = __BlockSize_LLT;

				_L2.set(L, k, j, l, n);
				_L.set(L, k, i, l, m);
				__times_A_BT_LLT(_L2, _L1, _L);
			}
		}
	}
	for (i = 0; i < L.cols; ++i) {
		__set0(L[i], i);
	}
}

template <typename t1, typename t2>
void LLT(MAT<t1>& A, MAT<t2>& L) {
	for (int i = 0; i < A.cols; ++i) {
		__copy(L[i] + i, A[i] + i, A.rows - i);
	}
	__LLT(L);
}

//==============================================================================================================
//==============================================================================================================

#ifdef XiangZiMatImport_SIMD
void __times_A_BT_LDLT(__BlockMat<double>& A, __BlockMat<double>& B, __BlockMat<double>& C, double* d) {
	int i, j;

	for (j = 0; j < C.cols; j = j + 8) {
		if (j + 8 > C.cols) {
			int k;
			for (; j < C.cols; ++j) {
				for (k = 0; k < A.cols; ++k) {
					__mixed(C[j], A[k], -B(j, k) * d[j], C[j], C.rows);
				}
			}
			break;
		}
		else {
			for (i = 0; i < C.rows; i = i + 8) {
				if (i + 8 > C.rows) {
					int y, k;
					for (y = j; y < j + 8; ++y) {
						for (k = 0; k < A.cols; ++k) {
							__mixed(C[y] + i, A[k] + i, -B(y, k) * d[k], C[y] + i, C.rows - i);
						}
					}
				}
				else __micro_kernel_8x8_ntimes_ADBT(A.data + i, A._rows, B.data + j, B._rows, C[j] + i, C._rows, A.cols, d);
			}
		}
	}
}
void __times_A_BT_LDLT2(__BlockMat<double>& A, __BlockMat<double>& C, double* d) {
	int i, j;

	for (j = 0; j < C.cols; j = j + 8) {
		for (i = j; i < C.rows; i = i + 8) {
			if (i + 8 > C.rows) {
				int y, k, n;
				if (j + 8 > C.cols) n = C.cols;
				else n = j + 8;

				for (y = j; y < n; ++y) {
					for (k = 0; k < A.cols; ++k) {
						__mixed(C[y] + i, A[k] + i, -A(y, k) * d[k], C[y] + i, C.rows - i);
					}
				}
			}
			else __micro_kernel_8x8_ntimes_ADBT(A.data + i, A._rows, A.data + j, A._rows, C[j] + i, C._rows, A.cols, d);
		}
	}
}
#endif

template<typename t>
void __times_A_BT_LDLT(__BlockMat<t>& A, __BlockMat<t>& B, __BlockMat<t>& C, t* d) {
	int i, j;

	for (j = 0; j < C.cols; j = j + 4) {
		if (j + 4 > C.cols) {
			int k;
			for (; j < C.cols; ++j) {
				for (k = 0; k < A.cols; ++k) {
					__mixed(C[j], A[k], -B(j, k) * d[j], C[j], C.rows);
				}
			}
			break;
		}
		else {
			for (i = 0; i < C.rows; i = i + 4) {
				if (i + 4 > C.rows) {
					int y, k;
					for (y = j; y < j + 4; ++y) {
						for (k = 0; k < A.cols; ++k) {
							__mixed(C[y] + i, A[k] + i, -B(y, k) * d[k], C[y] + i, C.rows - i);
						}
					}
				}
				else __micro_kernel_4x4_ntimes_ADBT(A.data + i, A._rows, B.data + j, B._rows, C[j] + i, C._rows, A.cols, d);
			}
		}
	}
}
template<typename t>
void __times_A_BT_LDLT2(__BlockMat<t>& A, __BlockMat<t>& C, t* d) {
	int i, j;

	for (j = 0; j < C.cols; j = j + 4) {
		for (i = j; i < C.rows; i = i + 4) {
			if (i + 4 > C.rows) {
				int y, k, n;
				if (j + 4 > C.cols) n = C.cols;
				else n = j + 4;

				for (y = j; y < n; ++y) {
					for (k = 0; k < A.cols; ++k) {
						__mixed(C[y] + i, A[k] + i, -A(y, k) * d[k], C[y] + i, C.rows - i);
					}
				}
			}
			else __micro_kernel_4x4_ntimes_ADBT(A.data + i, A._rows, A.data + j, A._rows, C[j] + i, C._rows, A.cols, d);
		}
	}
}

template<typename T>
void __Solve_direct_XDLT(__BlockMat<T>& L, __BlockMat<T>& X, T* d) {
	T* x;
	int j, k;
	for (j = 0; j < X.cols; ++j) {
		x = X[j];
		__times(x, (T)1 / L(j, j), x, X.rows);
		for (k = j + 1; k < X.cols; ++k) {
			__mixed(X[k], x, -L(k, j), X[k], X.rows);
		}
	}
	for (j = 0; j < X.cols; ++j) {
		__times(X[j], (T)1 / d[j], X[j], X.rows);
	}
}
template <typename t>
void __LDLT2(__BlockMat<t>& L, t* d) {
	t temp;
	int i, j;
	for (j = 0; j < L.cols; ++j) {
		d[j] = L(j, j);
		temp = (t)1 / d[j];
		__times(L[j] + j, temp, L[j] + j, L.rows - j);
		for (i = j + 1; i < L.rows; ++i) {
			__mixed(L[i] + i, L[j] + i, -L(i, j) * d[j], L[i] + i, L.rows - i);
		}
	}
}
template <typename t>
void __LDLT(MAT<t>& L, t* d) {
	int i, j, k;
	int m, n, l;
	__BlockMat<t> _L, _L1, _L2;

	for (j = 0; j < L.cols; j = j + __BlockSize_LLT) {
		if (j + __BlockSize_LLT > L.cols) n = L.cols - j;
		else n = __BlockSize_LLT;

		_L.set(L, j, j, n, n);
		__LDLT2(_L, d + j);

		for (i = j + __BlockSize_LLT; i < L.cols; i = i + __BlockSize_LLT) {
			if (i + __BlockSize_LLT > L.cols) m = L.cols - i;
			else m = __BlockSize_LLT;
			_L1.set(L, i, j, m, n);
			__Solve_direct_XDLT(_L, _L1, d + j);
		}

		for (i = j + __BlockSize_LLT; i < L.cols; i = i + __BlockSize_LLT) {
			if (i + __BlockSize_LLT > L.cols) m = L.cols - i;
			else m = __BlockSize_LLT;

			//k = i
			_L1.set(L, i, j, m, n);
			_L.set(L, i, i, m, m);
			__times_A_BT_LDLT2(_L1, _L, d + j);

			for (k = i + __BlockSize_LLT; k < L.cols; k = k + __BlockSize_LLT) {
				if (k + __BlockSize_LLT > L.cols) l = L.cols - k;
				else l = __BlockSize_LLT;

				_L2.set(L, k, j, l, n);
				_L.set(L, k, i, l, m);
				__times_A_BT_LDLT(_L2, _L1, _L, d + j);
			}
		}
	}
	for (i = 0; i < L.cols; ++i) {
		__set0(L[i], i);
	}
}

template <typename t1, typename t2>
void LDLT(MAT<t1>& A, MAT<t2>& L, MAT<t2>& D) {

	t2* d = D.data;
	__set0(d, D.rows * D.cols);
	for (int i = 0; i < A.cols; ++i) {
		__copy(L[i] + i, A[i] + i, A.rows - i);
	}

	__LDLT(L, d);

	for (int i = 1; i < D.cols; ++i) {
		D(i, i) = d[i];
	}
	__set0(d + 1, D.cols);
}
