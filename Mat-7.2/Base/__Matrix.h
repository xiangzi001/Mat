
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
		memcpy(data, a.data, sizeof(T) * s);
	}
	MAT(MAT&& a) :
		rows(a.rows),
		cols(a.cols),
		data(a.data)
	{
		a.data = 0;
	}

	~MAT() {
		if (data != 0) {
			delete[] data;
			data = 0;
		}
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
	void fill(T a) {
		int s = rows * cols;
		for (int i = 0; i < s; ++i)
			data[i] = a;
	}

	T* operator [] (int c) {
		return (data + rows * c);
	}
	T& operator () (int r, int c) {
		return data[rows * c + r];
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

template<typename t>
MAT<t> _T(const MAT<t>& A) {
	MAT<t> n(A.cols, A.rows);
	int i, j;
	for (i = 0; i < n.rows; ++i)
		for (j = 0; j < n.cols; ++j) {
			n.data[j * n.rows + i] = A.data[i * A.rows + j];
		}

	return n;
}

template<typename t>
MAT<complex<t>> _H(const MAT<complex<t>>& A) {
	MAT<complex<t>> n(A.cols, A.rows);
	int i, j;
	for (i = 0; i < n.rows; ++i)
		for (j = 0; j < n.cols; ++j) {
			n.data[j * n.rows + i] = A.data[i * A.rows + j].conj();
		}

	return n;
}

