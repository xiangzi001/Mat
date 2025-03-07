
#include"__Base.h"
#include"__QR_1.h"
#include"__QR_2.h"
#include"__Jacobi.h"

template<typename T>
void EVD_Q(MAT<T>& A, MAT<T>& E, MAT<T>& Q, double eps, EVD_type type) {
	switch (type)
	{
	case EVD_QR: {
		for (int i = 0; i < E.rows * E.cols; ++i) {
			E.data[i] = A.data[i];
		}
		__EVD_Q_QR_1(E, Q, eps);
		break;
	}
	case EVD_QR_2: {
		for (int i = 0; i < E.rows * E.cols; ++i) {
			E.data[i] = A.data[i];
		}
		__EVD_Q_QR_2(E, Q, eps);
		break;
	}
	case EVD_Jacobi: {
		for (int i = 0; i < E.rows * E.cols; ++i) {
			E.data[i] = A.data[i];
		}
		__EVD_Q_Jacobi(E, Q, eps);
		break;
	}
	default:
		break;
	}
}


