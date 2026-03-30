
template <typename T>
class __BlockMat {
public:
	int rows;
	int cols;
	int _rows;
	int _start;
	T* data;
	__BlockMat(const MAT<T>& A, int m, int n, int r, int c) :
		rows(r),
		cols(c),
		_rows(A.rows),
		_start(m),
		data(A.data + A.rows * n)
	{
	}
	T* operator [] (int c) {
		return (data + _rows * c + _start);
	}
	T& operator () (int r, int c) {
		return data[_rows * c + _start + r];
	}
};
