
#include"__QR.h"
#include"__Jacobi.h"
#include"__Eigen.h"

//Thin A(r, c) U(r, s) S(s, s) V(c, s)
//Full  A(r, c) U(r, r) S(r, c) V(c, c)
template<typename t1, typename t2>
void SVD(MAT<t1>& A, MAT<t2>& U, MAT<t2>& S, MAT<t2>& V, double eps, SVD_type type) {
	switch (type) {
	case SVD_QR: {
		__copy(S.data, A.data, A.rows * A.cols);
		__SVD_QR(U, S, V, eps);
		break;
	}
	case SVD_Jacobi: {
		__copy(S.data, A.data, A.rows * A.cols);
		__SVD_Jacobi(U, S, V, eps);
		break;
	}
	case SVD_Eigen_QR: {
		__SVD_EVD_QR(A, U, S, V, eps);
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
