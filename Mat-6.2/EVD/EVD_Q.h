
#include"__Base.h"
#include"__QR_1.h"
#include"__QR_2.h"
#include"__QR_A.h"
#include"__Jacobi.h"

template<typename T>
void EVD_Q(MAT<T>& A, MAT<T>& E, MAT<T>& Q, double eps, EVD_type type) {
	memcpy(E.data, A.data, sizeof(T) * E.rows * E.cols);
	switch (type)
	{
	case EVD_QR_1: {
		__EVD_Q_QR_1(E, Q, eps);
		break;
	}
	case EVD_QR_2: {
		__EVD_Q_QR_2(E, Q, eps);
		break;
	}
	case EVD_QR_A: {
		__EVD_Q_QR_A(E, Q, eps);
		break;
	}
	case EVD_Jacobi: {
		__EVD_Q_Jacobi(E, Q, eps);
		break;
	}
	default:
		break;
	}
}


