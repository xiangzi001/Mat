//*
template<typename T>
void __Solve_direct_XU(__BlockMat<T>& U, __BlockMat<T>& X) {
    int i;
    for (int r = 0; r < X.cols; ++r) {
        for (i = r + 1; i < U.cols; ++i) {
            __mixed(X[i], X[r], -U(r, i), X[i], X.rows);
        }
    }
}

template<typename T>
void __Solve_direct_XL(__BlockMat<T>& L, __BlockMat<T>& X) {
    int i;
    for (int r = X.cols - 1; r > 0; --r) {
        for (i = 0; i < r; ++i) {
            __mixed(X[i], X[r], -L(r, i), X[i], X.rows);
        }
    }
}

template <typename T>
void __Inv_X_LU(MAT<T>& A, MAT<T>& X, T* __restrict A_pack, T* __restrict B_pack) {
    int r, i, j, l, m, n;
    __BlockMat<T> _L, _U, _A;

    constexpr int MC = __BlockSize_Mul_MC;
    constexpr int NC = __BlockSize_Mul_NC;
    constexpr int K = __BlockSize_LU;

    for (r = 0; r < A.cols; r += K) {
        l = (r + K > A.cols) ? (A.cols - r) : K;

        _U.set(A, r, r, l, l);

        for (i = 0; i <= r; i += K) {
            m = (i + K > X.rows) ? (X.rows - i) : K;

            _A.set(X, i, r, m, l);
            __Solve_direct_XU(_U, _A);
        }

        int rows_to_pack = (r + l > X.rows) ? X.rows : (r + l);
        _L.set(X, 0, r, rows_to_pack, l);
        __pack_A_block<T, T>(_L.data, _L._rows, rows_to_pack, l, A_pack);

        for (j = r + K; j < X.cols; j += NC) {
            n = (j + NC > X.cols) ? (X.cols - j) : NC;

            _U.set(A, r, j, l, n);
            __pack_B_block<T, T>(_U[0], _U._rows, l, n, B_pack);

            for (i = 0; i < rows_to_pack; i += MC) {
                m = (i + MC > rows_to_pack) ? (rows_to_pack - i) : MC;

                _A.set(X, i, j, m, n);
                constexpr int MR = __micro_kernel_mr<T, T, T>();
                const T* A_base = A_pack + (size_t)(i / MR) * (size_t)l * MR;
                __mulAB_packed_n<T>(A_base, B_pack, _A.data, _A._rows,
                    _A.rows, _A.cols, l);
            }
        }
    }

    for (r = ((X.rows - 1) / K) * K; r >= 0; r -= K) {
        n = (r + K < X.rows) ? K : (X.rows - r);

        _L.set(A, r, r, n, n);

        for (i = 0; i < A.rows; i += K) {
            m = (i + K > X.rows) ? (X.rows - i) : K;
            _A.set(X, i, r, m, n);
            __Solve_direct_XL(_L, _A);
        }

        if (r > 0) {
            _U.set(X, 0, r, X.rows, n);
            __pack_A_block<T, T>(_U.data, _U._rows, X.rows, n, A_pack);
        }

        for (j = 0; j < r; j += NC) {
            int jc = (j + NC > r) ? (r - j) : NC;

            _L.set(A, r, j, n, jc);
            __pack_B_block<T, T>(_L[0], _L._rows, n, jc, B_pack);

            for (i = 0; i < X.rows; i += MC) {
                m = (i + MC > X.rows) ? (X.rows - i) : MC;

                _A.set(X, i, j, m, jc);
                constexpr int MR = __micro_kernel_mr<T, T, T>();
                const T* A_base = A_pack + (size_t)(i / MR) * (size_t)n * MR;
                __mulAB_packed_n<T>(A_base, B_pack, _A.data, _A._rows,
                    _A.rows, _A.cols, n);
            }
        }
    }
}

template<typename T>
void __Inv_swap(MAT<T>& A, int i, int* pos, int j, bool* flag) {
    int k = pos[j];
    if (k != i) {
        __Inv_swap(A, i, pos, k, flag);
    }
    __copy(A[k], A[j], A.rows);
    flag[j] = false;
}

template<typename T>
MAT<T> Inv(MAT<T> A) {
    MAT<T> B(A.rows, A.cols);
    int i;

    constexpr int NC = __BlockSize_Mul_NC;
    constexpr int K = __BlockSize_LU;
    constexpr int MR = __micro_kernel_mr<T, T, T>();
    constexpr int NR = __micro_kernel_nr<T, T, T>();

    const size_t a_pack_size = (size_t)((A.rows + MR - 1) / MR) * K * MR;
    const size_t b_pack_size = (size_t)((NC + NR - 1) / NR) * K * NR;

    T* A_pack = (T*)_mm_malloc(sizeof(T) * a_pack_size, 64);
    T* B_pack = (T*)_mm_malloc(sizeof(T) * b_pack_size, 64);

    int* pos = new int[B.rows];
    for (i = 0; i < A.rows; ++i) {
        pos[i] = i;
    }

    __PLU(pos, A, A, A_pack, B_pack);

    __set0(B.data, B.rows * B.cols);
    T temp;
    for (i = 0; i < A.cols; ++i) {
        temp = (T)1 / A(i, i);
        __times(A[i], temp, A[i], i);
        B(i, i) = temp;
    }

    __Inv_X_LU(A, B, A_pack, B_pack);

    bool* if_nswap = (bool*)A_pack;
    T* Temp = (T*)(if_nswap + B.cols);

    for (i = 0; i < B.cols; ++i) {
        if_nswap[i] = true;
    }
    int j;
    for (i = 0; i < B.cols; ++i) {
        if (if_nswap[i]) {
            j = pos[i];
            if (i != j) {
                __copy(Temp, B[i], B.rows);
                __Inv_swap(B, i, pos, i, if_nswap);
                __copy(B[j], Temp, B.rows);
            }
        }
    }
    delete[] pos;

    _mm_free(A_pack);
    _mm_free(B_pack);

    return B;
}
//*/
