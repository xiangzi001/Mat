
#include"__Base.h"
#include"__QR_1.h"
#include"__QR_2.h"
#include"__Jacobi.h"
#include"__Eigen.h"

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

//THIN A(r, c) U(r, s) S(s, s) V(c, s)
//FULL  A(r, c) U(r, r) S(r, c) V(c, c)
template<typename T>
void SVD(MAT<T>& A, MAT<T>& U, MAT<T>& S, MAT<T>& V, double eps, SVD_type type) {
	switch (type) {
	case SVD_QR: {
		__SVD_QR_1(A, U, S, V, eps);
		break;
	}
	case SVD_QR_2: {
		__SVD_QR_2(A, U, S, V, eps);
		break;
	}
	case SVD_Jacobi: {
		__SVD_Jacobi(A, U, S, V, eps);
		break;
	}
	//case _SVD_Eigen_QR_full: {
	//	SVD_EVD_QR_full(A, U, S, V, eps);
	//	break;
	//}
	case SVD_Eigen_QR: {
		__SVD_EVD_QR_1(A, U, S, V, eps);
		break;
	}
	case SVD_Eigen_QR_2: {
		__SVD_EVD_QR_2(A, U, S, V, eps);
		break;
	}
	case SVD_Eigen_Jacobi: {
		__SVD_EVD_Jacobi(A, U, S, V, eps);
		break;
	}
	default:
		break;
	}
}
