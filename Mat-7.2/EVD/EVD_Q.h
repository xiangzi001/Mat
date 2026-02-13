
#include"__QR.h"
#include"__Jacobi.h"

template<typename t1, typename t2>
void EVD_Q(MAT<t1>& A, MAT<t2>& E, MAT<t2>& Q, double eps, EVD_type type) {
	__copy(E.data, A.data, E.rows * E.cols);
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
	default:
		break;
	}
}


