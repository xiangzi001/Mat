
/*
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
		rows(r), cols(c)
	{
		//data = (T*)alloca(sizeof(T) * r * c);
	}

	template<class t>
	MAT(int r, int c, t* p) :
		rows(r), cols(c)
	{
		int s = r * c;
		data = (T*)alloca(s * sizeof(T));
		for (int i = 0; i < s; ++i) {
			data[i] = p[i];
		}
	}

	void init(int r, int c) {
		rows = r;
		cols = c;
		data = (T*)alloca(sizeof(T) * r * c);
	}
	template<typename t>
	void init(int r, int c, t* p) {
		rows = r;
		cols = c;
		int s = r * c;
		data = (T*)alloca(s * sizeof(T));
		for (int i = 0; i < s; ++i) {
			data[i] = p[i];
		}
	}
	void iden(int a) {
		rows = a;
		cols = a;
		int s = a * a;
		data = (T*)alloca(s * sizeof(T));
		for (int i = 0; i < s; ++i)
			data[i] = 0;
		for (int i = 0; i < a; ++i)
			data[i * (a + 1)] = 1;
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

	MAT operator * (MAT b) {
		T v;
		MAT n(rows, b.cols);
		n.fill(0);
		for (int i = 0; i < n.rows; ++i) {
			for (int k = 0; k < cols; ++k) {
				v = data[i * cols + k];
				for (int j = 0; j < n.cols; ++j)
					n[i][j] += v * b[k][j];
			}
		}
		b.data = 0;
		return n;
	}

	MAT operator + (MAT b) {
		MAT n(rows, cols);
		int s = n.rows * n.cols;
		for (int x = 0; x < s; ++x) {
			n.data[x] = data[x] + b.data[x];
		}
		b.data = 0;
		return n;
	}

	MAT operator - (MAT b) {
		MAT n(rows, cols);
		int s = n.rows * n.cols;
		for (int x = 0; x < s; ++x) {
			n.data[x] = data[x] - b.data[x];
		}
		b.data = 0;
		return n;
	}

	//copy B to A
	void operator = (MAT B) {
		delete[] data;
		rows = B.rows;
		cols = B.cols;
		int s = rows * cols;
		data = (T*)alloca(s * sizeof(T));
		for (int i = 0; i < s; ++i) {
			data[i] = B.data[i];
		}
		B.data = 0;
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
	//put A to B
	void operator >> (MAT& B) {
		B.rows = rows;
		rows = -1;
		B.cols = cols;
		cols = -1;
		B.data = data;
		data = 0;
	}
	//put B to A
	void operator << (MAT& B) {
		rows = B.rows;
		B.rows = -1;
		cols = B.cols;
		B.cols = -1;
		data = B.data;
		B.data = 0;
	}
};
*/
template <typename T>
class MatData {
public:
	int rows;
	int cols;
	T* data;

	MatData() :
		data(0), rows(-1), cols(-1)
	{
	}
	MatData(int r, int c) :
		data(new T[r * c]), rows(r), cols(c)
	{
	}
	template<class t>
	MatData(int r, int c, t* p) :
		rows(r), cols(c)
	{
		int s = r * c;
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
	T* operator [] (int r) {
		return (data + cols * r);
	}
};

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
	MAT(MatData<T> a) :
		rows(a.rows),
		cols(a.cols),
		data(a.data)
	{
	}

	~MAT() {
		delete[] data;
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

	MatData<T> operator * (T a) {
		MatData n(rows, cols);
		int s = rows * cols;
		for (int i = 0; i < s; ++i)
			n.data[i] = data[i]*a;
		return n;
	}

	MatData<T> operator * (MAT& b) {
		T v;
		MatData<T> n(rows, b.cols);
		int i, j, k;
		k = n.rows * n.cols;
		for (i = 0; i < k; ++i) {
			n.data[i] = 0;
		}
		for (i = 0; i < n.rows; ++i) {
			for (k = 0; k < cols; ++k) {
				v = data[i * cols + k];
				for (j = 0; j < n.cols; ++j)
					n[i][j] += v * b[k][j];
			}
		}
		
		return n;
	}

	MatData<T> operator + (MAT& b) {
		MatData<T> n(rows, cols);
		int s = n.rows * n.cols;
		for (int x = 0; x < s; ++x) {
			n.data[x] = data[x] + b.data[x];
		}
		
		return n;
	}

	MatData<T> operator - (MAT& b) {
		MatData<T> n(rows, cols);
		int s = n.rows * n.cols;
		for (int x = 0; x < s; ++x) {
				n.data[x] = data[x] - b.data[x];
			}
		
		return n;
	}

	//copy B to A
	void operator = (MAT& B) {
		delete[] data;
		rows = B.rows;
		cols = B.cols;
		int s = rows * cols;
		data = new T[s];
		for (int i = 0; i < s; ++i) {
			data[i] = B.data[i];
		}
	}
	//copy B to A
	void operator = (MatData<T> B) {
		delete[] data;
		rows = B.rows;
		cols = B.cols;
		data = B.data;
	}
	//copy B to A
	template <class t>
	void operator = (MatData<t> B) {
		delete[] data;
		rows = B.rows;
		cols = B.cols;
		int s = rows * cols;
		data = new T[s];
		for (int i = 0; i < s; ++i) {
			data[i] = B.data[i];
		}
		delete[] B.data;
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
	//put A to B
	void operator >> (MAT& B) {
		B.rows = rows;
		rows = -1;
		B.cols = cols;
		cols = -1;
		delete[] B.data;
		B.data = data;
		data = 0;
	}
	//put B to A
	void operator << (MAT& B) {
		rows = B.rows;
		B.rows = -1;
		cols = B.cols;
		B.cols = -1;
		delete[] data;
		data = B.data;
		B.data = 0;
	}
};
//*/

template <class t1, class t2>
MatData<t1> operator * (MatData<t1>a, MatData<t2> b) {
	MAT<t1> A(a);
	MAT<t2> B(b);

	return A * B;
}
template <class t1, class t2>
MatData<t1> operator * (MAT<t1>&a, MatData<t2> b) {
	MAT<t2> B(b);

	return a * B;
}
template <class t1, class t2>
MatData<t1> operator * (MatData<t1>a, MAT<t2>& b) {
	MAT<t1> A(a);

	return A * b;
}

template <class t1, class t2>
MatData<t1> operator + (MatData<t1>a, MatData<t2> b) {
	MAT<t1> A(a);
	MAT<t2> B(b);

	return A + B;
}
template <class t1, class t2>
MatData<t1> operator + (MAT<t1>& a, MatData<t2> b) {
	MAT<t2> B(b);

	return a + B;
}
template <class t1, class t2>
MatData<t1> operator + (MatData<t1>a, MAT<t2>& b) {
	MAT<t1> A(a);

	return A + b;
}

template <class t1, class t2>
MatData<t1> operator - (MatData<t1>a, MatData<t2> b) {
	MAT<t1> A(a);
	MAT<t2> B(b);

	return A - B;
}
template <class t1, class t2>
MatData<t1> operator - (MAT<t1>& a, MatData<t2> b) {
	MAT<t2> B(b);

	return a - B;
}
template <class t1, class t2>
MatData<t1> operator - (MatData<t1>a, MAT<t2>& b) {
	MAT<t1> A(a);

	return A - b;
}

enum QR_type {
	_QR_HouseHolder,
	_QR_Givens
};

enum EVD_type {
	_EVD_QR,
	_EVD_Jacobi
};

enum SVD_type {
	//full QR
	// A(r, c) U(r, r) S(r, c) V(c, c)
	_SVD_QR,

	//full Jacobi
	// A(r, c) U(r, r) S(r, c) V(c, c)
	_SVD_Jacobi,

	//full EVD_QR
	// A(r, c) U(r, r) S(r, c) V(c, c)
	_SVD_Eigen_QR_full,

	//thin EVD_QR
	// A(r, c) U(r, s) S(s, s) V(c, s)
	_SVD_Eigen_QR_thin,

	//thin EVD_Jacobi
	// A(r, c) U(r, s) S(s, s) V(c, s)
	_SVD_Eigen_Jacobi,

	_SVD_dqds
};


#include "Function.h"
#include "Complex.h"
#include "Base.h"
#include "Norm.h"
#include "Solve.h"
#include "LU.h"
#include "QR.h"
#include "QR_c.h"
#include "LLT.h"
#include "Schur.h"
#include "EVD_Q.h"
#include "EVD_X.h"
#include "EigenValue.h"
#include "SVD.h"
#include "SingularValue.h"
