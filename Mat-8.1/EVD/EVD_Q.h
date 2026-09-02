
#include"__QR.h"
#include"__Jacobi.h"

#include"__CDC.h"
#include"__ADC.h"

template<typename t1, typename t2>
inline void EVD_Q(MAT<t1>& A, MAT<t2>& E, MAT<t2>& Q, t2 eps, EVD_type type) {
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
	case EVD_CDC: {
		__EVD_Q_CDC(E, Q, eps);
		break;
	}
	case EVD_ADC: {
		__EVD_Q_ADC(E, Q, eps);
		break;
	}
	default:
		break;
	}
}


