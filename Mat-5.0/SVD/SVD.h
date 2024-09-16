
#include"__Base.h"
#include"__QR_1.h"
#include"__QR_2.h"
#include"__Jacobi.h"
#include"__Eigen.h"

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
