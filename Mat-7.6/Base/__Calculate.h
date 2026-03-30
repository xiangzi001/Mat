
template<typename T>
inline void __times(T* __restrict a, T b, T* __restrict ans, int len) {
	for (int i = 0; i < len; ++i) {
		ans[i] = a[i] * b;
	}
}

template<typename T>
inline void __add(T* __restrict a, T b, T* __restrict ans, int len) {
	for (int i = 0; i < len; ++i) {
		ans[i] = a[i] + b;
	}
}
template<typename T>
inline void __add(T* __restrict a, T* __restrict b, T* __restrict ans, int len) {
	for (int i = 0; i < len; ++i) {
		ans[i] = a[i] + b[i];
	}
}
template<typename t>
inline void __add(complex<t>* a, complex<t>* b, complex<t>* ans, int len) {
	__add((t*)a, (t*)b, (t*)ans, len * 2);
}

template<typename T>
inline void __sub(T* __restrict a, T* __restrict b, T* __restrict ans, int len) {
	for (int i = 0; i < len; ++i) {
		ans[i] = a[i] - b[i];
	}
}
template<typename t>
inline void __sub(complex<t>* a, complex<t>* b, complex<t>* ans, int len) {
	__sub((t*)a, (t*)b, (t*)ans, len * 2);
}

//ans=a+b*p
template<typename T>
inline void __mixed(T* __restrict a, T* __restrict b, T q, T* __restrict ans, int len) {
	for (int i = 0; i < len; ++i) {
		ans[i] = a[i] + b[i] * q;
	}
}
//ans=a+b*p
template<typename t1, typename t2, typename t3, typename t4>
inline void __mixed(t1* __restrict a, t2* __restrict b, t3 q, t4* __restrict ans, int len) {
	for (int i = 0; i < len; ++i) {
		ans[i] = a[i] + b[i] * q;
	}
}
template<typename T>
inline void __swap(T* __restrict a, T* __restrict b, int len) {
	T temp;
	for (int i = 0; i < len; ++i) {
		temp = a[i];
		a[i] = b[i];
		b[i] = temp;
	}
}
/*
template<typename T>
inline T __dot(T* __restrict a, T* __restrict b, int len) {
	T ans(0);
	for (int i = 0; i < len; ++i) {
		ans += a[i] * b[i];
	}
	return ans;
}
*/
template<typename T>
inline T __dot(T* __restrict a, T* __restrict b, int len) {
	T temp[4] = {};
	int n = 4 * ((int)(len / 4));
	for (int i = 0; i < n; i += 4) {
		temp[0] += a[i] * b[i];
		temp[1] += a[i + 1] * b[i + 1];
		temp[2] += a[i + 2] * b[i + 2];
		temp[3] += a[i + 3] * b[i + 3];
	}
	T ans = temp[0] + temp[1] + temp[2] + temp[3];
	for (int i = n; i < len; ++i) {
		ans += a[i] * b[i];
	}
	return ans;
}

template<typename t>
inline complex<t> __dot(complex<t>* __restrict a, complex<t>* __restrict b, int len) {
	complex<t> ans(0);
	for (int i = 0; i < len; ++i) {
		ans += a[i].conj() * b[i];
	}
	return ans;
}

template<typename t1, typename t2, typename t3>
void times_A_B(const MAT<t1>& A, const MAT<t2>& B, const MAT<t3>& C) {
	int j, k;
	__set0(C.data, C.rows * C.cols);
	for (j = 0; j < C.cols; ++j) {
		for (k = 0; k < A.cols; ++k) {
			__mixed(C.data + C.rows * j, A.data + A.rows * k, (B.data + B.rows * j)[k], C.data + C.rows * j, C.rows);
		}
	}
}

template<typename t1, typename t2, typename t3>
void times_A_BT(const MAT<t1>& A, const MAT<t2>& B, const MAT<t3>& C) {
	int j, k;
	__set0(C.data, C.rows * C.cols);
	for (j = 0; j < C.cols; ++j) {
		for (k = 0; k < A.cols; ++k) {
			__mixed(C.data + j * C.rows, A.data + k * A.rows, (B.data + k * B.rows)[j], C.data + j * C.rows, C.rows);
		}
	}
}
template<typename t1, typename t2, typename t3>
void times_A_BH(const MAT<t1>& A, const MAT<complex<t2>>& B, const MAT<t3>& C) {
	int j, k;
	__set0(C.data, C.rows * C.cols);
	for (j = 0; j < C.cols; ++j) {
		for (k = 0; k < A.cols; ++k) {
			__mixed(C.data + j * C.rows, A.data + k * A.rows, (B.data + k * B.rows)[j].conj(), C.data + j * C.rows, C.rows);
		}
	}
}

template<typename t1, typename t2, typename t3>
void times_AT_B(const MAT<t1>& A, const MAT<t2>& B, const MAT<t3>& C) {
	int i, j;
	t3* Temp;
	for (j = 0; j < C.cols; ++j) {
		Temp = C.data + j * C.rows;
		for (i = 0; i < C.rows; ++i) {
			Temp[i] = __dot(A.data + i * A.rows, B.data + j * B.rows, A.rows);
		}
	}
}

template<class t>
MAT<t> operator * (t a, const MAT<t>& A) {
	MAT<t> n(A.rows, A.cols);

	__times(A.data, a, n.data, A.rows * A.cols);

	return n;
}
template<class t>
MAT<t> operator * (const MAT<t>& A, t a) {
	MAT<t> n(A.rows, A.cols);

	__times(A.data, a, n.data, A.rows * A.cols);

	return n;
}
template<class t>
MAT<t> operator / (const MAT<t>& A, t a) {
	MAT<t> n(A.rows, A.cols);

	__times(A.data, t(1) / a, n.data, A.rows * A.cols);

	return n;
}
/*
template<typename t1, typename t2>
auto operator * (const MAT<t1>& A, const __TemplateTranspose<t2>& B) {
	MAT<decltype(std::declval<t1>()* std::declval<t2>())> n(A.rows, B.__Mat.rows);
	times_A_BT(A, B.__Mat, n);
	return n;
}
template<typename t1, typename t2>
auto operator * (const __TemplateTranspose<t1>& A, const MAT<t2>& B) {
	MAT<decltype(std::declval<t1>()* std::declval<t2>())> n(A.__Mat.cols, B.cols);
	times_AT_B(A.__Mat, B, n);
	return n;
}
template<typename t1, typename t2>
auto operator * (const MAT<t1>& A, const __TemplateDagger<t2>& B) {
	MAT<decltype(std::declval<t1>()* std::declval<t2>())> n(A.rows, B.__Mat.rows);
	times_A_BH(A, B.__Mat, n);
	return n;
}
template<typename t1, typename t2>
auto operator * (const __TemplateDagger<t1>& A, const MAT<t2>& B) {
	MAT<decltype(std::declval<t1>()* std::declval<t2>())> n(A.__Mat.cols, B.cols);
	times_AT_B(A.__Mat, B, n);
	return n;
}
*/
template<typename t1, typename t2>
auto operator * (const MAT<t1>& A, const MAT<t2>& B) {
	MAT<decltype(std::declval<t1>() * std::declval<t2>())> n(A.rows, B.cols);
	times_A_B(A, B, n);
	return n;
}
template<typename t1, typename t2>
auto operator + (const MAT<t1>& A, const MAT<t2>& B) {
	MAT<decltype(std::declval<t1>() + std::declval<t2>())> n(A.rows, A.cols);
	__add(A.data, B.data, n.data, A.rows * A.cols);
	return n;
}
template<typename t1, typename t2>
auto operator - (const MAT<t1>& A, const MAT<t2>& B) {
	MAT<decltype(std::declval<t1>() - std::declval<t2>())> n(A.rows, A.cols);
	__sub(A.data, B.data, n.data, A.rows * A.cols);
	return n;
}
