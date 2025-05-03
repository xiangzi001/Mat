
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
        int s = A.cols * A.rows;
        int i;
        memcpy(R.data, A.data, sizeof(T) * s);
        s = Q.cols * Q.rows;
        for (i = 0; i < s; ++i) {
            Q.data[i] = 0;
        }
        for (i = 0; i < Q.rows; ++i) {
            Q[i][i] = 1;
        }

        T* w = new T[R.rows + 64 / sizeof(T) - 1];
        T* v = new T[R.cols + 64 / sizeof(T) - 1];
        __QR_HouseHolder(Q, R, w, v);
        delete[] w;
        delete[] v;
        break;
    }
    case QR_Givens: {
        int s = A.cols * A.rows;
        int i;
        for (i = 0; i < s; ++i) {
            R.data[i] = A.data[i];
        }
        s = Q.cols * Q.rows;
        for (i = 0; i < s; ++i) {
            Q.data[i] = 0;
        }
        for (i = 0; i < Q.rows; ++i) {
            Q[i][i] = 1;
        }
        __QR_Givens(Q, R);
        break;
    }
    case QR_GramSchmidt: {
        int s = A.cols * A.rows;
        int i,j;
        for (i = 0; i < s; ++i) {
            R.data[i] = 0;
        }
        for (i = 0; i < Q.rows; ++i) {
            for (j = 0; j < Q.cols; ++j) {
                Q[i][j] = A[i][j];
            }
        }
        __QR_GramSchmidt(Q, R);
        break;
    }
    default:
        break;
    }
}

