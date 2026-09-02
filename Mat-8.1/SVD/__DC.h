

template<typename T>
void __SVD_DC(MAT<T>& U, MAT<T>& S, MAT<T>& V, T eps) {
	int i = 0;
	int r;

	T* v;
	if (S.rows > S.cols) {

		v = new T[S.rows];
		__SVD_Householder_Up(S, U, V, v);
		delete[] v;

		T* a = S.data + (S.rows - 1) * S.cols;
		T* b = a - S.cols;

		r = S.cols - 1;
		__copy(b, b + 1, r);
		b[r] = 0;
		__SVD_Givens_UP(a, b, U, V, 0, r, eps);

		__set0(S.data, (S.rows - 1) * S.cols);
		T temp = a[r];
		for (i = 0; i < r; ++i) {
			S(i, i) = a[i];
		}
		__set0(a, S.cols);
		S(r, r) = temp;

	}
	else {
		v = new T[S.cols];
		__SVD_Householder_Down(S, U, V, v);
		delete[] v;

		T* a = S.data + (S.cols - 1) * S.rows;
		T* b = a - S.rows;

		r = S.rows - 1;
		__copy(b, b + 1, r);
		__SVD_Givens_UP(a, b, V, U, 0, r, eps);

		__set0(S.data, (S.cols - 1) * S.rows);
		T temp = a[r];
		for (i = 0; i < r; ++i) {
			S(i, i) = a[i];
		}
		__set0(a, S.rows);
		S(r, r) = temp;
	}
}
