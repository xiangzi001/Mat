
//
// 
// void random(MAT<T>& A, int r, int c, int Big, int Small)
// void random(MAT<complex<T>>& A, int r, int c, int Big, int Small)
// MatData<t> _T(MAT<t>& A)
// MatData<complex<t>> _H(MAT<complex<t>>& A)
// 
// double Norm_one(MAT<T>& A)
// double Norm_inf(MAT<T>& A)
// T Norm_F(MAT<T>& A)
// 
// T Det(MAT<T>& n)
// void Solve_direct(MAT<T>& A, MAT<T>& b)
// MatData<T> Inv(MAT<T>& A)
// int Rank(MAT<T>& A, double eps = 0)
// 
// void LU(MAT<T>& A, MAT<T>& L, MAT<T>& U)
// void LLT(MAT<T>& A, MAT<T>& L)
// void LDLT(MAT<T>& A, MAT<T>& L, MAT<T>& D)
// 
// void QR(MAT<T>& A, MAT<T>& Q, MAT<T>& R, QR_type type)
// void QR(MAT<complex<T>>& A, MAT<complex<T>>& Q, MAT<complex<T>>& R, QR_type type)
// 
// 
// void Solve_Jacobi(MAT<T>& A, T* b, T*& x, T eps)
// void Solve_GS(MAT<T>& A, T* b, T* x, T eps)
// void Solve_SOR(MAT<T>& A, T* b, T* x, T eps, T w = 0.1)
// void Solve_Krylov_SD(MAT<T>& A, T* b, T* x, T eps)
// void Solve_Krylov_CG(MAT<T>& A, T* b, T* x, T eps)
// 
// void Schur_R(MAT<t>& A, MAT<t>& T, MAT<t>& U, double eps)
// void Schur_I(MAT<complex<t1>>& A, MAT<complex<t2>>& T, MAT<complex<t2>>& U, double eps)
// 
// void EVD_Q(MAT<T>& A, MAT<T>& E, MAT<T>& Q, double eps, EVD_type type)
// void EVD_X(MAT<t1>& A, MAT<complex<t2>>& E, MAT<complex<t2>>& X, double eps)
// 
// void EV(MAT<t1>& A, MAT<complex<t2>>& E, double eps)
// void Roots(Func<_t>& func, complex<t>* ans, double eps)
// 
// void SVD(MAT<T>& A, MAT<T>& U, MAT<T>& S, MAT<T>& V, double eps, SVD_type type)
// 
// void SV(MAT<T>& A, MAT<T>& S, double eps, SV_type type)
// 
// 


#ifndef XiangZiMatImport
#define XiangZiMatImport

enum QR_type {
	QR_HouseHolder,
	QR_Givens,
	QR_GramSchmidt
};
enum EVD_type {
	EVD_QR,
	EVD_QR_2,//先集中迭代特征值 再求特征向量 ( n >= 4 )
	EVD_Jacobi
};
enum SV_type {
	SV_QR,
	SV_dqds
};
enum SVD_type {
	//full QR
	// A(r, c) U(r, r) S(r, c) V(c, c)
	SVD_QR,
	SVD_QR_2,//先集中迭代奇异 再求U,V ( n >= 4 )

	//full Jacobi
	// A(r, c) U(r, r) S(r, c) V(c, c)
	SVD_Jacobi,

	//thin EVD_QR
	// A(r, c) U(r, s) S(s, s) V(c, s)
	SVD_Eigen_QR,
	SVD_Eigen_QR_2,//先集中迭代特征值 再求特征向量 ( n >= 4 )

	//thin EVD_Jacobi
	// A(r, c) U(r, s) S(s, s) V(c, s)
	SVD_Eigen_Jacobi
};

#include <immintrin.h>

#include "Base/Base.h"

#include "LU/LU.h"
#include "QR/QR.h"

#include "Schur/Schur.h"

#include "EV/EV.h"
#include "EVD/EVD_Q.h"
#include "EVD/EVD_X.h"

#include "SV/SV.h"
#include "SVD/SVD.h"

#include "Geometry/Geometry.h"

#endif