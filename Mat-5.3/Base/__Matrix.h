

template <typename T>
class MAT {
public:
	int rows;
	int cols;
	T* data;

	MAT() :
		data(0), rows(-1), cols(-1)
	{
	}

	MAT(int r, int c) :
		data(new T[r * c]), rows(r), cols(c)
	{
	}

	template<class t>
	MAT(int r, int c, t* p) :
		rows(r), cols(c)
	{
		int s = r * c;
		data = new T[s];
		for (int i = 0; i < s; ++i) {
			data[i] = p[i];
		}
	}
	MAT(MAT& a) :
		rows(a.rows),
		cols(a.cols)
	{
		int s = rows * cols;
		data = new T[s];
		for (int i = 0; i < s; ++i) {
			data[i] = a.data[i];
		}
	}
	MAT(MAT&& a) :
		rows(a.rows),
		cols(a.cols),
		data(a.data)
	{
		a.data = 0;
	}

	~MAT() {
		delete[] data;
		data = 0;
	}

	void init(int r, int c) {
		rows = r;
		cols = c;
		delete[] data;
		data = new T[r * c];
	}
	template<typename t>
	void init(int r, int c, t* p) {
		rows = r;
		cols = c;
		int s = r * c;
		delete[] data;
		data = new T[s];
		for (int i = 0; i < s; ++i) {
			data[i] = p[i];
		}
	}
	void iden(int a) {
		rows = a;
		cols = a;
		int s = a * a;
		delete[] data;
		data = new T[s];
		for (int i = 0; i < s; ++i)
			data[i] = 0;
		for (int i = 0; i < a; ++i)
			data[i * cols + i] = 1;
	}
	void fill(int a) {
		int s = rows * cols;
		for (int i = 0; i < s; ++i)
			data[i] = a;
	}

	T* operator [] (int r) {
		return (data + cols * r);
	}

	MAT operator * (T a) {
		MAT n(rows, cols);
		int s = rows * cols;
		for (int i = 0; i < s; ++i)
			n.data[i] = data[i] * a;
		return n;
	}

	template <class t>
	MAT operator * (const MAT<t>& b) {
		T v;
		MAT n(rows, b.cols);
		int i, j, k;
		k = n.rows * n.cols;
		for (i = 0; i < k; ++i) {
			n.data[i] = 0;
		}
		for (i = 0; i < n.rows; ++i) {
			for (k = 0; k < cols; ++k) {
				v = data[i * cols + k];
				for (j = 0; j < n.cols; ++j)
					n[i][j] += v * b.data[k * b.cols + j];
			}
		}

		return n;
	}

	template <class t>
	MAT operator + (const MAT<t>& b) {
		MAT n(rows, cols);
		int s = n.rows * n.cols;
		for (int x = 0; x < s; ++x) {
			n.data[x] = data[x] + b.data[x];
		}

		return n;
	}

	template <class t>
	MAT operator - (const MAT<t>& b) {
		MAT n(rows, cols);
		int s = n.rows * n.cols;
		for (int x = 0; x < s; ++x) {
			n.data[x] = data[x] - b.data[x];
		}

		return n;
	}

	//copy B to A
	void operator = (MAT<T> B) {
		rows = B.rows;
		cols = B.cols;
		T* temp = B.data;
		B.data = data;
		data = temp;
	}
	template <class t>
	void operator = (const MAT<t>& B) {
		rows = B.rows;
		cols = B.cols;
		int s = rows * cols;
		delete[] data;
		data = new T[s];
		for (int i = 0; i < s; ++i) {
			data[i] = B.data[i];
		}
	}

	//exchange A and B
	void operator & (MAT& B) {
		int temp = rows;
		rows = B.rows;
		B.rows = temp;

		temp = cols;
		cols = B.cols;
		B.cols = temp;

		T* _temp = data;
		data = B.data;
		B.data = _temp;
	}

	template <typename U>
	friend class VEC;
};

template<class t>
MAT<t> operator * (t a, const MAT<t>& A) {
	MAT<t> n(A.rows, A.cols);
	int s = A.rows * A.cols;
	for (int i = 0; i < s; ++i)
		n.data[i] = A.data[i] * a;
	return n;
}

template<typename t>
MAT<t> _T(const MAT<t>& A) {
	MAT<t> n(A.cols, A.rows);
	for (int i = 0; i < A.cols; ++i)
		for (int j = 0; j < A.rows; ++j) {
			n[i][j] = A.data[j * A.cols + i];
		}

	return n;
}

template<typename t>
MAT<complex<t>> _H(const MAT<complex<t>>& A) {
	MAT<complex<t>> n(A.cols, A.rows);
	for (int i = 0; i < A.cols; ++i)
		for (int j = 0; j < A.rows; ++j) {
			n[i][j] = A.data[j * A.cols + i].conj();
		}

	return n;
}

template<typename T>
void times(MAT<T>& A, MAT<T>& B, MAT<T>& C) {
	int i, j, k = C.rows * C.cols;
	for (i = 0; i < k; ++i) {
		C.data[i] = 0;
	}
	T v;
	for (i = 0; i < C.rows; ++i) {
		for (k = 0; k < A.cols; ++k) {
			v = A[i][k];
			for (j = 0; j < C.cols; ++j)
				C[i][j] += v * B[k][j];
		}
	}
}

template<typename T>
void times_A_BT(MAT<T>& A, MAT<T>& B, MAT<T>& C) {
	int i, j, k;
	for (i = 0; i < C.rows; ++i) {
		for (j = 0; j < C.rows; ++j) {
			C[i][j] = 0;
			for (k = 0; k < A.cols; ++k)
				C[i][j] += A[i][k] * B[j][k];
		}
	}
}

template<typename T>
void times_AT_B(MAT<T>& A, MAT<T>& B, MAT<T>& C) {
	int i, j, k = C.rows * C.cols;
	for (i = 0; i < k; ++i)
		C.data[i] = 0;
	for (k = 0; k < A.rows; ++k) {
		for (i = 0; i < C.rows; ++i) {
			for (j = 0; j < C.cols; ++j) {
				C[i][j] += A[k][i] * B[k][j];
			}
		}
	}
}
