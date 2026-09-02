
#ifndef XiangZiMatImport
#define XiangZiMatImport

#include <cmath>

//
// 
// void random(MAT<T>& A, int r, int c, int Big, int Small)
// void random(MAT<complex<T>>& A, int r, int c, int Big, int Small)
// MAT<t> _T(MAT<t>& A)
// MAT<complex<t>> _H(MAT<complex<t>>& A)
// 
// double Norm_one(MAT<T>& A)
// double Norm_inf(MAT<T>& A)
// T Norm_F(MAT<T>& A)
// 
// T Det(MAT<T>& n)
// void Solve_direct(MAT<T>& A, MAT<T>& B)
// MAT<T> Inv(MAT<T>& A)
// 
// void LU(MAT<t1>& A, MAT<t2>& L, MAT<t2>& U)
// void LLT(MAT<T>& A, MAT<T>& L)
// void LDLT(MAT<T>& A, MAT<T>& L, MAT<T>& D)
//  
// void QR(MAT<t1>& A, MAT<t2>& Q, MAT<t2>& R, QR_type type)
// void QR(MAT<t1>& A, MAT<complex<t2>>& Q, MAT<complex<t2>>& R, QR_type type)
// 
// void Schur_R(MAT<t1>& A, MAT<t2>& T, MAT<t2>& U, t2 eps)
// void Schur_I(MAT<t1>& A, MAT<complex<t2>>& T, MAT<complex<t2>>& U, t2 eps)
// 
// void EVD_Q(MAT<t1>& A, MAT<t2>& E, MAT<t2>& Q, t2 eps, EVD_type type)
// void EVD_X(MAT<t1>& A, MAT<complex<t2>>& E, MAT<complex<t2>>& X, t2 eps)
// void EVD_X(MAT<complex<t1>>& A, MAT<complex<t2>>& E, MAT<complex<t2>>& X, t2 eps)
// 
// void EV(MAT<t1>& A, complex<t2>* E, t2 eps)
// void EV(MAT<complex<t1>>& A, complex<t2>* E, t2 eps)
// void Roots(Func<t1>& func, complex<t2>* ans, t2 eps)
// 
// void SVD(MAT<t1>& A, MAT<t2>& U, MAT<t2>& S, MAT<t2>& V, t2 eps, SVD_type type)
// 

enum QR_type {
	QR_HouseHolder,
	QR_Givens,
	QR_GramSchmidt
};
enum EVD_type {
	EVD_QR,
	EVD_Jacobi
};
//Thin A(r, c) U(r, s) S(s, s) V(c, s)
//Full  A(r, c) U(r, r) S(r, c) V(c, c)
enum SVD_type {
	SVD_QR,//full
	SVD_Jacobi,//full
};

#include "SIMD/SIMD.h"
#include "Base/Base.h"
#include "D:\[1]xiao xiang\d. c++\{5}math\math-Mat\Math\_extra\extra.h"

#include "LU/LU.h"

#include "QR/QR.h"

#include "Schur/Schur.h"

#include "EV/EV.h"
#include "EVD/EVD_Q.h"
#include "EVD/EVD_X.h"

#include "SVD/SVD.h"

#include "Geometry/Geometry.h"

template<typename t1, typename t2>
void LU_init(MAT<t1>& A, MAT<t2>& L, MAT<t2>& U) {
	L.init(A.rows, A.rows);
	U.init(A.rows, A.cols);
}
template<typename t1, typename t2>
void QR_init(MAT<t1>& A, MAT<t2>& Q, MAT<t2>& R) {
	Q.init(A.rows, A.rows);
	R.init(A.rows, A.cols);
}
template<typename t1, typename t2>
void EVD_init(MAT<t1>& A, MAT<t2>& E, MAT<t2>& X) {
	E.init(A.rows, A.rows);
	X.init(A.rows, A.rows);
}
template<typename t1, typename t2>
void SVD_init(MAT<t1>& A, MAT<t2>& U, MAT<t2>& S, MAT<t2>& V) {
	U.init(A.rows, A.rows);
	S.init(A.rows, A.cols);
	V.init(A.cols, A.cols);
}
#endif