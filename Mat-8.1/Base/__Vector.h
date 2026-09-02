

template <typename T>
class VEC : public MAT<T>
{
public:
	VEC() {
		this->rows = 0;
		this->cols = 1;
		this->data = nullptr;
	}

	VEC(int len) : MAT<T>(len, 1) {
	}

	template<typename t>
	VEC(int len, t* p) : MAT<T>(len, 1, p) {
	}
	template<typename t>
	VEC(int len, t value, ...) : MAT<T>(len, 1) {
		va_list num;
		va_start(num, value);
		this->data[0] = value;
		for (int i = 1; i < len; ++i) {
			this->data[i] = va_arg(num, t);
		}
		va_end(num);
	}

	VEC(VEC& a) : MAT<T>(a.rows, 1, a.data) {
	}
	VEC(VEC&& a) {
		this->rows = a.rows;
		this->cols = 1;
		this->data = a.data;
		a.data = 0;
	}
	VEC(MAT<T>& A) {
		this->rows = A.rows * A.cols;
		this->cols = 1;
		this->data = new T[this->rows];
		__copy(this->data, A.data, A.rows * A.cols);
	}
	VEC(MAT<T>&& A) {
		this->rows = A.rows * A.cols;
		this->cols = 1;
		this->data = A.data;
		A.data = 0;
	}

	void init(int r) {
		if (this->rows != r) {
			this->rows = r;
			this->cols = 1;
			delete[] this->data;
			this->data = new T[r];
		}
	}
	template<typename t>
	void init(int r, t* p) {
		if (this->rows != r) {
			this->rows = r;
			this->cols = 1;
			delete[] this->data;
			this->data = new T[r];
		}
		for (int i = 0; i < r; ++i) {
			this->data[i] = p[i];
		}
	}
	template<typename t>
	void init(int r, t value, ...) {
		if (this->rows != r) {
			this->rows = r;
			this->cols = 1;
			delete[] this->data;
			this->data = new T[r];
		}
		va_list num;
		va_start(num, value);
		this->data[0] = value;
		for (int i = 1; i < r; ++i) {
			this->data[i] = va_arg(num, t);
		}
		va_end(num);
	}


	void operator = (VEC<T> x) {
		this->rows = x.rows;
		this->cols = 1;
		T* temp = x.data;
		x.data = this->data;
		this->data = temp;
	}
	template <class t>
	void operator = (const VEC<t>& x) {
		if (this->rows != x.rows) {
			this->rows = x.rows;
			delete[] this->data;
			this->data = new T[this->rows];
		}
		__copy(this->data, x.data, this->rows);
	}

	template <class t>
	void operator = (const MAT<t>& A) {
		if (this->rows != A.rows * A.cols) {
			this->rows = A.rows * A.cols;
			delete[] this->data;
			this->data = new T[this->rows];
		}
		__copy(this->data, A.data, this->rows);
	}

	T& operator [](int n) {
		return (this->data)[n];
	}
	T& operator ()(int n) {
		return (this->data)[n];
	}
};
