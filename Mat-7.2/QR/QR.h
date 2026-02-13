
#include"__HouseHolder.h"
#include"__Givens.h"
#include"__GramSchmidt.h"

template<typename t1, typename t2>
void QR(MAT<t1>& A, MAT<t2>& Q, MAT<t2>& R, QR_type type) {
    switch (type)
    {
    case QR_HouseHolder: {
        int i;
        __copy(R.data, A.data, A.cols * A.rows);
        ZeroMemory(Q.data, sizeof(t2) * Q.cols * Q.rows);
        for (i = 0; i < Q.rows; ++i) {
            Q[i][i] = 1;
        }

        t2* w = new t2[R.rows];
        t2* v = new t2[R.cols];
        __QR_HouseHolder(Q, R, w, v);
        delete[] w;
        delete[] v;
        break;
    }
    case QR_Givens: {
        int i;
        __copy(R.data, A.data, A.cols * A.rows);
        ZeroMemory(Q.data, sizeof(t2) * Q.cols * Q.rows);
        for (i = 0; i < Q.rows; ++i) {
            Q[i][i] = 1;
        }
        __QR_Givens(Q, R);
        break;
    }
    case QR_GramSchmidt: {
        ZeroMemory(R.data, sizeof(t2) * R.cols * R.rows);
        __copy(Q.data, A.data, Q.cols * Q.rows);
        __QR_GramSchmidt(Q, R);
        break;
    }
    default:
        break;
    }
}

