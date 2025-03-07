
#include"__QR.h"
#include"__dqds.h"

template<typename T>
void SV(MAT<T>& A, MAT<T>& S, double eps, SV_type type) {
	switch (type) {
	case SV_QR:{
		__SV_QR(A, S, eps);
		break;
	}
	case SV_dqds: {
		__SV_dqds(A, S, eps);
		break;
	}
	default:
		break;
	}
}
