
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
	_QR_Givens,
	_QR_GramSchmidt
};
template<typename T>
void QR_init(MAT<T>& A, MAT<T>& Q, MAT<T>& R) {
	Q.init(A.rows, A.rows);
	R.init(A.rows, A.cols);
}

enum EVD_type {
	_EVD_QR_1,//集中三对角迭代特征值
	_EVD_QR_2,//先集中迭代特征值 再求特征向量
	_EVD_Jacobi
};
template<typename T>
void EVD_init(MAT<T>& A, MAT<T>& E, MAT<T>& X) {
	E.init(A.rows, A.rows);
	X.init(A.rows, A.rows);
}

enum SVD_type {
	//full QR
	// A(r, c) U(r, r) S(r, c) V(c, c)
	_SVD_QR,

	//full Jacobi
	// A(r, c) U(r, r) S(r, c) V(c, c)
	_SVD_Jacobi,

	//thin EVD_QR
	// A(r, c) U(r, s) S(s, s) V(c, s)
	_SVD_Eigen_QR_1,//集中三对角迭代特征值
	_SVD_Eigen_QR_2,//先集中迭代特征值 再求特征向量

	//thin EVD_Jacobi
	// A(r, c) U(r, s) S(s, s) V(c, s)
	_SVD_Eigen_Jacobi
};
template<typename T>
void SVD_init_full(MAT<T>& A, MAT<T>& U, MAT<T>& S, MAT<T>& V) {
	U.init(A.rows, A.rows);
	S.init(A.rows, A.cols);
	V.init(A.cols, A.cols);
}
template<typename T>
void SVD_init_thin(MAT<T>& A, MAT<T>& U, MAT<T>& S, MAT<T>& V) {
	int s;
	if (A.rows > A.cols) {
		s = A.cols;
	}
	else {
		s = A.rows;
	}
	U.init(A.rows, s);
	S.init(s, s);
	V.init(A.cols, s);
}
//
// 
// void random(MAT<T>& A, int r, int c, int Big, int Small)
// void random(MAT<complex<T>>& A, int r, int c, int Big, int Small)
// MatData<t> _T(MAT<t>& A)
// MatData<complex<t>> _H(MAT<complex<t>>& A)
// 
// double Norm_one(MAT<T>& A)
// double Norm_inf(MAT<T>& A)
// T Norm_F(MAT<T>& A)
// 
// T det(MAT<T>& n)
// void solve(MAT<T>& A, MAT<T>& b)
// void solve_c(MAT<T>& A, MAT<T>& b)
// MatData<T> inv(MAT<T>& A)
// MatData<T> inv_c(MAT<T>& A)
// int Rank(MAT<T>& A, double eps = 0)
// int Rank_c(MAT<T>& A, double eps = 0)
// 
// void LU(MAT<T>& A, MAT<T>& L, MAT<T>& U)
// void LLT(MAT<T>& A, MAT<T>& L)
// void LDLT(MAT<T>& A, MAT<T>& L, MAT<T>& D)
// void LDLT_2(MAT<T>& A, MAT<T>& L, MAT<T>& D)
// 
// void QR(MAT<T>& A, MAT<T>& Q, MAT<T>& R, QR_type type)
// void QR(MAT<complex<T>>& A, MAT<complex<T>>& Q, MAT<complex<T>>& R, QR_type type)
// 
// void Solve(Func<_t>& func, complex<t>* ans, double eps)
// void EVD_Q(MAT<T>& A, MAT<T>& E, MAT<T>& Q, double eps, EVD_type type)
// void EVD_X(MAT<complex<t>>& A, MAT<complex<t>>& E, MAT<complex<t>>& X, double eps)
// void Schur(MAT<complex<t>>& A, MAT<complex<t>>& T, MAT<complex<t>>& U, double eps)
// 
// void SVD(MAT<T>& A, MAT<T>& U, MAT<T>& S, MAT<T>& V, double eps, SVD_type type)
// 
// 





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

#include "_File.h"
#include "_Print.h"


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