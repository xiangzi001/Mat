
template <typename T>
void LU(MAT<T>& A, MAT<T>& L, MAT<T>& U) {
	memcpy(U.data, A.data, sizeof(T) * U.rows * U.cols);
	ZeroMemory(L.data, sizeof(T) * U.rows * U.cols);
	for (int i = 0; i < L.rows; ++i) {
		L[i][i] = 1;
	}
}
