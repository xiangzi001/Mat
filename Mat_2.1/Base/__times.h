
#define __BlockSize_times 64

template<typename t1, typename t2, typename t3>
void __times_A_B(__BlockMat<t1>& A, __BlockMat<t2>& B, __BlockMat<t3>& C) {
	int i, j;

	for (j = 0; j < C.cols; j = j + __micro_kernel_size<t1,t2,t3>()) {
		if (j + __micro_kernel_size<t1, t2, t3>() > C.cols) {
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
			for (i = 0; i < C.rows; i = i + __micro_kernel_size<t1, t2, t3>()) {
				if (i + __micro_kernel_size<t1, t2, t3>() > C.rows) {
					int y, k;
					for (y = j; y < j + __micro_kernel_size<t1, t2, t3>(); ++y) {
						t3* b = B[y];
						for (k = 0; k < A.cols; ++k) {
							__mixed(C[y] + i, A[k] + i, b[k], C[y] + i, C.rows - i);
						}
					}
				}
				else __micro_kernel_times_A_B(A.data + i, A._rows, B[j], B._rows, C[j] + i, C._rows, A.cols);
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
