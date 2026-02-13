
template <typename t1, typename t2>
void LLT(MAT<t1>& A, MAT<t2>& L) {
	t2 temp;
	int i, j;
	for (i = 0; i < A.cols; ++i) {
		ZeroMemory(L[i], sizeof(t2) * i);
		__copy(L[i] + i, A[i] + i, A.rows - i);
	}
	for (j = 0; j < A.cols; ++j) {
		temp = 1.0 / sqrt(L(j, j));
		__times(L[j] + j, temp, L[j] + j, L.rows - j);
		for (i = j + 1; i < A.rows; ++i) {
			__mixed3(L[i] + i, L[j] + i, -L(i, j), L[i] + i, L.rows - i);
		}
	}
}

template <typename t1, typename t2>
void LDLT(MAT<t1>& A, MAT<t2>& L, MAT<t2>& D) {
	t2 temp;
	t2* d = D.data;
	int i, j;

	for (i = 0; i < A.cols; ++i) {
		ZeroMemory(L[i], sizeof(t2) * i);
		__copy(L[i] + i, A[i] + i, A.rows - i);
	}
	ZeroMemory(d, sizeof(t2) * D.rows * D.cols);

	for (j = 0; j < A.cols; ++j) {
		d[j] = L(j, j);
		temp = 1.0 / d[j];
		__times(L[j] + j, temp, L[j] + j, L.rows - j);
		for (i = j + 1; i < A.rows; ++i) {
			__mixed3(L[i] + i, L[j] + i, -L(i, j) * d[j], L[i] + i, L.rows - i);
		}
	}
	for (i = 1; i < D.cols; ++i) {
		D(i, i) = d[i];
	}
	ZeroMemory(d + 1, sizeof(t2) * D.cols);
}
