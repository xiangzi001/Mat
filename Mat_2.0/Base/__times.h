
#define __BlockSize_times 128

#ifdef XiangZiMatImport_SIMD
void __times_A_B(__BlockMat<double>& A, __BlockMat<double>& B, __BlockMat<double>& C) {
	int i, j;

	for (j = 0; j < C.cols; j = j + 8) {
		if (j + 8 > C.cols) {
			int k;
			for (; j < C.cols; ++j) {
				double* b = B[j];
				for (k = 0; k < A.cols; ++k) {
					__mixed(C[j], A[k], b[k], C[j], C.rows);
				}
			}
			break;
		}
		else {
			for (i = 0; i < C.rows; i = i + 8) {
				if (i + 8 > C.rows) {
					int y, k;
					for (y = j; y < j + 8; ++y) {
						double* b = B[y];
						for (k = 0; k < A.cols; ++k) {
							__mixed(C[y] + i, A[k] + i, b[k], C[y] + i, C.rows - i);
						}
					}
				}
				else __micro_kernel_8x8_times_A_B(A.data + i, A._rows, B[j], B._rows, C[j] + i, C._rows, A.cols);
			}
		}
	}
}
#endif // XiangZiMatImport_SIMD

template<typename t1, typename t2, typename t3>
void __times_A_B(__BlockMat<t1>& A, __BlockMat<t2>& B, __BlockMat<t3>& C) {
	int i, j;

	for (j = 0; j < C.cols; j = j + 4) {
		if (j + 4 > C.cols) {
			int k;
			for (; j < C.cols; ++j) {
				t3* b = B[j];
				for (k = 0; k < A.cols; ++k) {
					__mixed(C[j], A[k], b[k], C[j], C.rows);
				}
			}
			break;
		}
		else {
			for (i = 0; i < C.rows; i = i + 4) {
				if (i + 4 > C.rows) {
					int y, k;
					for (y = j; y < j + 4; ++y) {
						t3* b = B[y];
						for (k = 0; k < A.cols; ++k) {
							__mixed(C[y] + i, A[k] + i, b[k], C[y] + i, C.rows - i);
						}
					}
				}
				else __micro_kernel_4x4_times_A_B(A.data + i, A._rows, B[j], B._rows, C[j] + i, C._rows, A.cols);
			}
		}
	}
}
template<typename t1, typename t2, typename t3>
void times_A_B(const MAT<t1>& A, const MAT<t2>& B, const MAT<t3>& C) {
	__BlockMat<t1> _A;
	__BlockMat<t2> _B;
	__BlockMat<t3> _C;
	int i, j, k;
	int l, m, n;
	__set0(C.data, C.rows * C.cols);
	for (k = 0; k < A.cols; k = k + __BlockSize_times) {
		if (k + __BlockSize_times > A.cols) l = A.cols - k;
		else l = __BlockSize_times;

		for (j = 0; j < C.cols; j = j + __BlockSize_times) {
			if (j + __BlockSize_times > C.cols) n = C.cols - j;
			else n = __BlockSize_times;

			for (i = 0; i < C.rows; i = i + __BlockSize_times) {
				if (i + __BlockSize_times > C.rows) m = C.rows - i;
				else m = __BlockSize_times;

				_A.set(A, i, k, m, l);
				_B.set(B, k, j, l, n);
				_C.set(C, i, j, m, n);
				__times_A_B(_A, _B, _C);
			}
		}
	}
}
template<typename t1, typename t2>
auto operator * (const MAT<t1>& A, const MAT<t2>& B) {
	MAT<decltype(std::declval<t1>()* std::declval<t2>())> n(A.rows, B.cols);
	times_A_B(A, B, n);
	return n;
}

/*
void __times_A_BT(__BlockMat<double>& A, __BlockMat<double>& B, __BlockMat<double>& C) {
	int i, j;

	for (j = 0; j < C.cols; j = j + 8) {
		if (j + 8 > C.cols) {
			int k;
			for (; j < C.cols; ++j) {
				for (k = 0; k < A.cols; ++k) {
					__mixed(C[j], A[k], B(j, k), C[j], C.rows);
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
							__mixed(C[y] + i, A[k] + i, B(y, k), C[y] + i, C.rows - i);
						}
					}
				}
				else __micro_kernel_8x8_times_A_BT(A.data + i, A._rows, B.data + j, B._rows, C[j] + i, C._rows, A.cols);
			}
		}
	}
}

template<typename t1, typename t2, typename t3>
void __times_A_BT2(__BlockMat<t1>& A, __BlockMat<t2>& B, __BlockMat<t3>& C) {
	int j, k;
	for (k = 0; k < A.cols; ++k){
		for (j = 0; j < C.cols; ++j) {
			__mixed(C[j], A[k], B(j, k), C[j], C.rows);
		}
	}
}
template<typename t1, typename t2, typename t3>
void __times_A_BT(__BlockMat<t1>& A, __BlockMat<t2>& B, __BlockMat<t3>& C) {
	int i, j;

	for (j = 0; j < C.cols; j = j + 4) {
		if (j + 4 > C.cols) {
			int k;
			for (; j < C.cols; ++j) {
				for (k = 0; k < A.cols; ++k) {
					__mixed(C[j], A[k], B(j, k), C[j], C.rows);
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
							__mixed(C[y] + i, A[k] + i, B(y, k), C[y] + i, C.rows - i);
						}
					}
				}
				else __micro_kernel_4x4_times_A_BT(A.data + i, A._rows, B.data + j, B._rows, C[j] + i, C._rows, A.cols);
			}
		}
	}
}
template<typename t1, typename t2, typename t3>
void times_A_BT(const MAT<t1>& A, const MAT<t2>& B, const MAT<t3>& C) {
	__BlockMat<t1> _A;
	__BlockMat<t2> _B;
	__BlockMat<t3> _C;
	int i, j, k;
	int l, m, n;
	__set0(C.data, C.rows * C.cols);
	for (k = 0; k < A.cols; k = k + __BlockSize_times) {
		if (k + __BlockSize_times > A.cols) l = A.cols - k;
		else l = __BlockSize_times;

		for (j = 0; j < C.cols; j = j + __BlockSize_times) {
			if (j + __BlockSize_times > C.cols) n = C.cols - j;
			else n = __BlockSize_times;

			for (i = 0; i < C.rows; i = i + __BlockSize_times) {
				if (i + __BlockSize_times > C.rows) m = C.rows - i;
				else m = __BlockSize_times;

				_A.set(A, i, k, m, l);
				_B.set(B, j, k, n, l);
				_C.set(C, i, j, m, n);
				__times_A_BT(_A, _B, _C);
			}
		}
	}
}

template<typename t1, typename t2, typename t3>
void __times_AT_B2(__BlockMat<t1>& A, __BlockMat<t2>& B, __BlockMat<t3>& C) {
	int i, j;
	t3* Temp;
	for (j = 0; j < C.cols; ++j) {
		Temp = C[j];
		for (i = 0; i < C.rows; ++i) {
			Temp[i] += __dot(A[i], B[j], A.rows);
		}
	}
}
template<typename t1, typename t2, typename t3>
void __times_AT_B(__BlockMat<t1>& A, __BlockMat<t2>& B, __BlockMat<t3>& C) {
	int i, j;
	t3* Temp;

	for (j = 0; j < C.cols; j = j + 4) {
		if (j + 4 > C.cols) {
			for (; j < C.cols; ++j) {
				Temp = C[j];
				for (i = 0; i < C.rows; ++i) {
					Temp[i] += __dot(A[i], B[j], A.rows);
				}
			}
			break;
		}
		else {
			for (i = 0; i < C.rows; i = i + 4) {
				if (i + 4 > C.rows) {
					int y, k;
					for (y = j; y < j + 4; ++y) {
						Temp = C[y];
						for (k = i; k < C.rows; ++k) {
							Temp[k] += __dot(A[k], B[y], A.rows);
						}
					}
				}
				else __micro_kernel_4x4_times_AT_B(A[i], A._rows, B[j], B._rows, C[j] + i, C._rows, A.rows);
			}
		}
	}
}
template<typename t1, typename t2, typename t3>
void times_AT_B(const MAT<t1>& A, const MAT<t2>& B, const MAT<t3>& C) {
	__BlockMat<t1> _A;
	__BlockMat<t2> _B;
	__BlockMat<t3> _C;
	int i, j, k;
	int l, m, n;
	__set0(C.data, C.rows * C.cols);
	for (j = 0; j < C.cols; j = j + __BlockSize_times) {
		if (j + __BlockSize_times > C.cols) n = C.cols - j;
		else n = __BlockSize_times;

		for (i = 0; i < C.rows; i = i + __BlockSize_times) {
			if (i + __BlockSize_times > C.rows) m = C.rows - i;
			else m = __BlockSize_times;

			for (k = 0; k < A.cols; k = k + __BlockSize_times) {
				if (k + __BlockSize_times > A.cols) l = A.cols - k;
				else l = __BlockSize_times;

				_A.set(A, k, i, l, m);
				_B.set(B, k, j, l, n);
				_C.set(C, i, j, m, n);
				__times_AT_B(_A, _B, _C);
			}
		}
	}
}

*/