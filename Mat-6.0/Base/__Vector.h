

template <typename T>
class VEC : public MAT<T>
{
public:
	VEC() : MAT<T>(){
	}

	VEC(int len) : MAT<T>(len , 1){
	}

	template<typename t>
	VEC(int len, t* p) : MAT<T>(len, 1, p){
	}
	template<typename t>
	VEC(int len, t value, ...) : MAT<T>(len, 1)
	{
		va_list num;
		va_start(num, value);
		this->data[0] = value;
		for (int i = 1; i < len; ++i) {
			this->data[i] = va_arg(num, t);
		}
		va_end(num);
	}

	VEC(VEC& a) : MAT<T>(a.rows, 1, a.data){
	}
	VEC(VEC&& a){
		this->rows = a.rows;
		this->cols = 1;
		this->data = a.data;
		a.data = 0;
	}
	VEC(MAT<T>& A){
		this->rows = A.rows * A.cols;
		this->cols = 1;
		delete[] this->data;
		this->data = new T[this->rows];
		int j;
		for (int i = 0; i < A.cols; ++i) {
			for (j = 0; j < A.rows; ++j) {
				this->data[i * A.cols + j] = A.data[j * A.cols + i];
			}
		}
	}
	VEC(MAT<T>&& A) {
		this->rows = A.rows * A.cols;
		this->cols = 1;
		T temp;
		int j;
		for (int i = 0; i < A.cols; ++i) {
			for (j = i + 1; j < A.rows; ++j) {
				temp = A.data[i * A.cols + j];
				A.data[i * A.cols + j] = A.data[j * A.cols + i];
				A.data[j * A.cols + i] = temp;
			}
		}
		this->data = A.data;
		A.data = 0;
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
		this->rows = x.rows;
		this->cols = x.cols;
		int s = this->rows * this->cols;
		delete[] this->data;
		this->data = new T[s];
		for (int i = 0; i < s; ++i) {
			this->data[i] = x.data[i];
		}
	}

	template <class t>
	void operator = (const MAT<t>& A) {
		this->rows = A.rows * A.cols;
		this->cols = 1;
		delete[] this->data;
		this->data = new T[this->rows];
		int j;
		for (int i = 0; i < A.cols; ++i) {
			for (j = 0; j < A.rows; ++j) {
				this->data[i * A.cols + j] = A.data[j * A.cols + i];
			}
		}
	}

	T& operator [](int n){
		return (this->data)[n];
	}
};

template<typename t>
t Length(const VEC<t>& x) {
	t ret(0);

	for (int i = 0; i < x.rows; ++i) {
		ret += x.data[i] * x.data[i];
	}

	return sqrt(ret);
}

template<typename t>
VEC<t> Column(const MAT<t>& A, int col) {
	VEC<t> ret(A.rows);

	for (int i = 0; i < A.rows; ++i) {
		ret[i] = A.data[i * A.cols + col];
	}

	return ret;
}