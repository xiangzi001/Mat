
#include"__HouseHolder.h"
#include"__Givens.h"
#include"__GramSchmidt.h"
#include"__HouseHolder_Blocked.h"

template<typename t1, typename t2>
inline void QR(MAT<t1>& A, MAT<t2>& Q, MAT<t2>& R, QR_type type) {
    switch (type)
    {
    case QR_Householder: {
        if ((void*)A.data != (void*)R.data)
            __copy(R.data, A.data, A.cols * A.rows);
        if (std::min(A.rows, A.cols) <= __BlockSize_QR * 2) {
            __QR_HouseHolder(Q, R);
        }
        else {
            using T = t2;
            const int m = R.rows, n = R.cols;
            __QR_Buffer<T> QR_buf = QR_WorkAlloc<T>(m, n);
            __QR_HouseHolder_Blocked(Q, R, QR_buf);
            QR_WorkFree<T>(QR_buf);
        }
        break;
    }
    case QR_Givens: {
        if ((void*)A.data != (void*)R.data)
            __copy(R.data, A.data, A.cols * A.rows);
        __set0(Q.data, Q.cols * Q.rows);
        for (int i = 0; i < Q.rows; ++i) {
            Q[i][i] = 1;
        }
        __QR_Givens(Q, R);
        break;
    }
    case QR_GramSchmidt: {
        __set0(R.data, R.cols * R.rows);
        if ((void*)A.data != (void*)Q.data)
            __copy(Q.data, A.data, Q.cols * Q.rows);
        __QR_GramSchmidt(Q, R);
        break;
    }
    default:
        break;
    }
}

template<QR_type type, typename t1, typename t2>
inline void QR(MAT<t1>& A, MAT<t2>& Q, MAT<t2>& R) {
    if constexpr (type == QR_Householder) {
        if ((void*)A.data != (void*)R.data)
            __copy(R.data, A.data, A.cols * A.rows);

        __QR_HouseHolder(Q, R);
    }
    else if constexpr (type == QR_Givens) {
        if ((void*)A.data != (void*)R.data)
            __copy(R.data, A.data, A.cols * A.rows);
        __set0(Q.data, Q.cols * Q.rows);
        for (int i = 0; i < Q.rows; ++i) {
            Q[i][i] = 1;
        }
        __QR_Givens(Q, R);
    }
    else if constexpr (type == QR_GramSchmidt) {
        __set0(R.data, R.cols * R.rows);
        if ((void*)A.data != (void*)Q.data)
            __copy(Q.data, A.data, Q.cols * Q.rows);
        __QR_GramSchmidt(Q, R);
    }
}

