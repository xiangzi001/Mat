
#include"__QR.h"
#include"__QR_DC.h"
#include"__Jacobi.h"

template<typename T>
void EVD_Q(MAT<T>& A, MAT<T>& E, MAT<T>& Q, double eps, EVD_type type) {
	memcpy(E.data, A.data, sizeof(T) * E.rows * E.cols);
	switch (type)
	{
	case EVD_QR: {
		__EVD_Q_QR(E, Q, eps);
		break;
	}
	case EVD_Jacobi: {
		__EVD_Q_Jacobi(E, Q, eps);
		break;
	}
	case EVD_QR_DC: {
		__EVD_Q_QR_DC(E, Q, eps);
		break;
	}
	default:
		break;
	}
}


