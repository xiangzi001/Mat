
#include"__HouseHolder.h"
#include"__Givens.h"
#include"__GramSchmidt.h"

template<typename T>
void QR_init(MAT<T>& A, MAT<T>& Q, MAT<T>& R) {
    Q.init(A.rows, A.rows);
    R.init(A.rows, A.cols);
}

template<typename T>
void QR(MAT<T>& A, MAT<T>& Q, MAT<T>& R, QR_type type) {
    switch (type)
    {
    case QR_HouseHolder: {
        int i;
        memcpy(R.data, A.data, sizeof(T) * A.cols * A.rows);
        ZeroMemory(Q.data, sizeof(T) * Q.cols * Q.rows);
        for (i = 0; i < Q.rows; ++i) {
            Q[i][i] = 1;
        }

        T* w = new T[R.rows];
        T* v = new T[R.cols];
        __QR_HouseHolder(Q, R, w, v);
        delete[] w;
        delete[] v;
        break;
    }
    case QR_Givens: {
        int i;
        memcpy(R.data, A.data, sizeof(T) * A.cols * A.rows);
        ZeroMemory(Q.data, sizeof(T) * Q.cols * Q.rows);
        for (i = 0; i < Q.rows; ++i) {
            Q[i][i] = 1;
        }
        __QR_Givens(Q, R);
        break;
    }
    case QR_GramSchmidt: {
        ZeroMemory(R.data, sizeof(T) * R.cols * R.rows);
        memcpy(Q.data, A.data, sizeof(T) * Q.cols * Q.rows);
        __QR_GramSchmidt(Q, R);
        break;
    }
    default:
        break;
    }
}

