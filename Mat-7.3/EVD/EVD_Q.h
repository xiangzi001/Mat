
#include"__QR.h"
#include"__Jacobi.h"

#include"__QR_CDC.h"

template<typename t1, typename t2>
inline void EVD_Q(MAT<t1>& A, MAT<t2>& E, MAT<t2>& Q, double eps, EVD_type type) {
	if(A.data != E.data)
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
	case EVD_QR_CDC: {
		__EVD_Q_QR_CDC(E, Q, eps);
		break;
	}
	default:
		break;
	}
}


