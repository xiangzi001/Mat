
#include"__Householder.h"
#include"__Givens.h"
#include"__GramSchmidt.h"

template<typename t1, typename t2>
inline void QR(MAT<t1>& A, MAT<t2>& Q, MAT<t2>& R, QR_type type) {
    switch (type)
    {
    case QR_Householder: {
        if (A.data != R.data)
            __copy(R.data, A.data, A.cols * A.rows);

        __QR_Householder(Q, R);
        break;
    }
    case QR_Givens: {
        int i;
        if (A.data != R.data)
            __copy(R.data, A.data, A.cols * A.rows);
        __set0(Q.data, Q.cols * Q.rows);
        for (i = 0; i < Q.rows; ++i) {
            Q[i][i] = 1;
        }
        __QR_Givens(Q, R);
        break;
    }
    case QR_GramSchmidt: {
        __set0(R.data, R.cols * R.rows);
        if (A.data != Q.data)
            __copy(Q.data, A.data, Q.cols * Q.rows);
        __QR_GramSchmidt(Q, R);
        break;
    }
    default:
        break;
    }
}

