template<typename t1, typename t2, typename t3>
constexpr int __micro_kernel_mr() {
    return 4;
}

template<typename t1, typename t2, typename t3>
constexpr int __micro_kernel_nr() {
    return 4;
}

template<typename t1, typename t2>
inline void __pack_A_block(
    const t1* __restrict A,
    size_t ldA, size_t mc, size_t kc,
    t1* __restrict Ap)
{
    constexpr size_t MR = 4;

    for (size_t ip = 0; ip * MR < mc; ++ip) {
        const size_t mm = std::min(MR, mc - ip * MR);
        t1* dst = Ap + ip * kc * MR;

        for (size_t p = 0; p < kc; ++p) {
            const t1* src = A + p * ldA + ip * MR;
            for (size_t i = 0; i < mm; ++i)
                dst[i] = src[i];
            for (size_t i = mm; i < MR; ++i)
                dst[i] = t1(0);
            dst += MR;
        }
    }
}
template<typename t1, typename t2>
inline void __pack_AT_block(
    const t1* __restrict Vd,
    size_t ldV, size_t pc,
    size_t mc, size_t kc,
    t1* __restrict Ap)
{
    constexpr size_t MR = 4;
    for (size_t ip = 0; ip * MR < mc; ++ip) {
        const size_t mm = std::min(MR, mc - ip * MR);
        t1* dst = Ap + ip * kc * MR;
        for (size_t q = 0; q < kc; ++q) {
            const t1* src = Vd + (ip * MR) * ldV + pc + q;

            for (size_t i = 0; i < mm; ++i)
                dst[i] = src[i * ldV];
            for (size_t i = mm; i < MR; ++i)
                dst[i] = 0;
            dst += MR;
        }
    }
}

template<typename t1, typename t2>
inline void __pack_B_block(
    const t2* __restrict B,
    size_t ldB, size_t kc, size_t nc,
    t2* __restrict Bp)
{
    constexpr size_t NR = 4;

    for (size_t jp = 0; jp * NR < nc; ++jp) {
        const size_t nn = std::min(NR, nc - jp * NR);
        t2* dst = Bp + jp * kc * NR;

        for (size_t p = 0; p < kc; ++p) {
            for (size_t j = 0; j < nn; ++j)
                dst[p * NR + j] = B[(jp * NR + j) * ldB + p];
            for (size_t j = nn; j < NR; ++j)
                dst[p * NR + j] = 0;
        }
    }
}

template<typename t1, typename t2>
inline void __pack_BT_block(
    const t2* __restrict B,
    size_t ldB, size_t kc, size_t nc,
    t2* __restrict Bp)
{
    constexpr int NR = 4;
    for (size_t jp = 0; jp * NR < nc; ++jp) {
        const int nn = (int)((nc - jp * NR < (size_t)NR) ? (nc - jp * NR) : (size_t)NR);
        t2* dst = Bp + jp * kc * NR;
        for (size_t p = 0; p < kc; ++p) {
            const t2* src = B + p * ldB + jp * NR;

            for (int j = 0; j < nn; ++j)
                dst[j] = src[j];
            for (int j = nn; j < NR; ++j)
                dst[j] = 0;
            dst += NR;
        }
    }
}

template<typename t1, typename t2>
inline void __pack_BT_block_d(
    const t2* __restrict B,
    size_t ldB, size_t kc, size_t nc,
    const t2* __restrict d,
    t2* __restrict Bp)
{
    constexpr int NR = 4;
    for (size_t jp = 0; jp * NR < nc; ++jp) {
        const int nn = (int)((nc - jp * NR < (size_t)NR) ? (nc - jp * NR) : (size_t)NR);
        t2* dst = Bp + jp * kc * NR;
        for (size_t p = 0; p < kc; ++p) {
            const t2 dp = d[p];
            const t2* src = B + p * ldB + jp * NR;

            for (int j = 0; j < nn; ++j)
                dst[j] = src[j] * dp;
            for (int j = nn; j < NR; ++j)
                dst[j] = 0;
            dst += NR;
        }
    }
}

template<typename t1, typename t2, typename t3>
static void __micro_kernel_mul_packed_full(
    const t1* __restrict Ap,
    const t2* __restrict Bp,
    t3* __restrict C,
    size_t ldC, size_t k)
{
    constexpr int MR = 4;
    constexpr int NR = 4;

    t3 c00 = C[0 * ldC + 0];
    t3 c10 = C[0 * ldC + 1];
    t3 c20 = C[0 * ldC + 2];
    t3 c30 = C[0 * ldC + 3];
    t3 c01 = C[1 * ldC + 0];
    t3 c11 = C[1 * ldC + 1];
    t3 c21 = C[1 * ldC + 2];
    t3 c31 = C[1 * ldC + 3];
    t3 c02 = C[2 * ldC + 0];
    t3 c12 = C[2 * ldC + 1];
    t3 c22 = C[2 * ldC + 2];
    t3 c32 = C[2 * ldC + 3];
    t3 c03 = C[3 * ldC + 0];
    t3 c13 = C[3 * ldC + 1];
    t3 c23 = C[3 * ldC + 2];
    t3 c33 = C[3 * ldC + 3];

    for (size_t p = 0; p < k; ++p) {
        const t1 a0 = Ap[p * MR + 0];
        const t1 a1 = Ap[p * MR + 1];
        const t1 a2 = Ap[p * MR + 2];
        const t1 a3 = Ap[p * MR + 3];
        const t2 b0 = Bp[p * NR + 0];
        const t2 b1 = Bp[p * NR + 1];
        const t2 b2 = Bp[p * NR + 2];
        const t2 b3 = Bp[p * NR + 3];

        c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
        c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
        c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
        c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
    }

    C[0 * ldC + 0] = c00;
    C[0 * ldC + 1] = c10;
    C[0 * ldC + 2] = c20;
    C[0 * ldC + 3] = c30;
    C[1 * ldC + 0] = c01;
    C[1 * ldC + 1] = c11;
    C[1 * ldC + 2] = c21;
    C[1 * ldC + 3] = c31;
    C[2 * ldC + 0] = c02;
    C[2 * ldC + 1] = c12;
    C[2 * ldC + 2] = c22;
    C[2 * ldC + 3] = c32;
    C[3 * ldC + 0] = c03;
    C[3 * ldC + 1] = c13;
    C[3 * ldC + 2] = c23;
    C[3 * ldC + 3] = c33;
}

template<typename t1, typename t2, typename t3>
static void __micro_kernel_mul_packed_edge(
    const t1* __restrict Ap,
    const t2* __restrict Bp,
    t3* __restrict C,
    size_t ldC, size_t k,
    int mr, int nr)
{
    constexpr int MR = 4;
    constexpr int NR = 4;

    t3 c00, c10, c20, c30;
    t3 c01, c11, c21, c31;
    t3 c02, c12, c22, c32;
    t3 c03, c13, c23, c33;

    if (nr > 0) {
        c00 = C[0 * ldC + 0];
        c10 = C[0 * ldC + 1];
        c20 = C[0 * ldC + 2];
        c30 = C[0 * ldC + 3];
    }
    if (nr > 1) {
        c01 = C[1 * ldC + 0];
        c11 = C[1 * ldC + 1];
        c21 = C[1 * ldC + 2];
        c31 = C[1 * ldC + 3];
    }
    if (nr > 2) {
        c02 = C[2 * ldC + 0];
        c12 = C[2 * ldC + 1];
        c22 = C[2 * ldC + 2];
        c32 = C[2 * ldC + 3];
    }
    if (nr > 3) {
        c03 = C[3 * ldC + 0];
        c13 = C[3 * ldC + 1];
        c23 = C[3 * ldC + 2];
        c33 = C[3 * ldC + 3];
    }

    for (size_t p = 0; p < k; ++p) {
        const t1 a0 = Ap[p * MR + 0];
        const t1 a1 = Ap[p * MR + 1];
        const t1 a2 = Ap[p * MR + 2];
        const t1 a3 = Ap[p * MR + 3];
        const t2 b0 = Bp[p * NR + 0];
        const t2 b1 = Bp[p * NR + 1];
        const t2 b2 = Bp[p * NR + 2];
        const t2 b3 = Bp[p * NR + 3];

        c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
        c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
        c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
        c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
    }

    if (nr > 0) {
        const t3* Cj = C + 0 * ldC;
        if (mr > 0) Cj[0] = c00;
        if (mr > 1) Cj[1] = c10;
        if (mr > 2) Cj[2] = c20;
        if (mr > 3) Cj[3] = c30;
    }
    if (nr > 1) {
        const t3* Cj = C + 1 * ldC;
        if (mr > 0) Cj[0] = c01;
        if (mr > 1) Cj[1] = c11;
        if (mr > 2) Cj[2] = c21;
        if (mr > 3) Cj[3] = c31;
    }
    if (nr > 2) {
        const t3* Cj = C + 2 * ldC;
        if (mr > 0) Cj[0] = c02;
        if (mr > 1) Cj[1] = c12;
        if (mr > 2) Cj[2] = c22;
        if (mr > 3) Cj[3] = c32;
    }
    if (nr > 3) {
        const t3* Cj = C + 3 * ldC;
        if (mr > 0) Cj[0] = c03;
        if (mr > 1) Cj[1] = c13;
        if (mr > 2) Cj[2] = c23;
        if (mr > 3) Cj[3] = c33;
    }
}


template<typename t1, typename t2, typename t3>
static void __micro_kernel_nmul_packed_full(
    const t1* __restrict Ap,
    const t2* __restrict Bp,
    t3* __restrict C,
    size_t ldC, size_t k)
{
    constexpr int MR = 4;
    constexpr int NR = 4;

    t3 c00 = C[0 * ldC + 0];
    t3 c10 = C[0 * ldC + 1];
    t3 c20 = C[0 * ldC + 2];
    t3 c30 = C[0 * ldC + 3];
    t3 c01 = C[1 * ldC + 0];
    t3 c11 = C[1 * ldC + 1];
    t3 c21 = C[1 * ldC + 2];
    t3 c31 = C[1 * ldC + 3];
    t3 c02 = C[2 * ldC + 0];
    t3 c12 = C[2 * ldC + 1];
    t3 c22 = C[2 * ldC + 2];
    t3 c32 = C[2 * ldC + 3];
    t3 c03 = C[3 * ldC + 0];
    t3 c13 = C[3 * ldC + 1];
    t3 c23 = C[3 * ldC + 2];
    t3 c33 = C[3 * ldC + 3];

    for (size_t p = 0; p < k; ++p) {
        const t1 a0 = Ap[p * MR + 0];
        const t1 a1 = Ap[p * MR + 1];
        const t1 a2 = Ap[p * MR + 2];
        const t1 a3 = Ap[p * MR + 3];
        const t2 b0 = Bp[p * NR + 0];
        const t2 b1 = Bp[p * NR + 1];
        const t2 b2 = Bp[p * NR + 2];
        const t2 b3 = Bp[p * NR + 3];

        c00 -= a0 * b0; c01 -= a0 * b1; c02 -= a0 * b2; c03 -= a0 * b3;
        c10 -= a1 * b0; c11 -= a1 * b1; c12 -= a1 * b2; c13 -= a1 * b3;
        c20 -= a2 * b0; c21 -= a2 * b1; c22 -= a2 * b2; c23 -= a2 * b3;
        c30 -= a3 * b0; c31 -= a3 * b1; c32 -= a3 * b2; c33 -= a3 * b3;
    }

    C[0 * ldC + 0] = c00;
    C[0 * ldC + 1] = c10;
    C[0 * ldC + 2] = c20;
    C[0 * ldC + 3] = c30;
    C[1 * ldC + 0] = c01;
    C[1 * ldC + 1] = c11;
    C[1 * ldC + 2] = c21;
    C[1 * ldC + 3] = c31;
    C[2 * ldC + 0] = c02;
    C[2 * ldC + 1] = c12;
    C[2 * ldC + 2] = c22;
    C[2 * ldC + 3] = c32;
    C[3 * ldC + 0] = c03;
    C[3 * ldC + 1] = c13;
    C[3 * ldC + 2] = c23;
    C[3 * ldC + 3] = c33;
}

template<typename t1, typename t2, typename t3>
static void __micro_kernel_nmul_packed_edge(
    const t1* __restrict Ap,
    const t2* __restrict Bp,
    t3* __restrict C,
    size_t ldC, size_t k,
    int mr, int nr)
{
    constexpr int MR = 4;
    constexpr int NR = 4;

    t3 c00, c10, c20, c30;
    t3 c01, c11, c21, c31;
    t3 c02, c12, c22, c32;
    t3 c03, c13, c23, c33;

    if (nr > 0) {
        c00 = C[0 * ldC + 0];
        c10 = C[0 * ldC + 1];
        c20 = C[0 * ldC + 2];
        c30 = C[0 * ldC + 3];
    }
    if (nr > 1) {
        c01 = C[1 * ldC + 0];
        c11 = C[1 * ldC + 1];
        c21 = C[1 * ldC + 2];
        c31 = C[1 * ldC + 3];
    }
    if (nr > 2) {
        c02 = C[2 * ldC + 0];
        c12 = C[2 * ldC + 1];
        c22 = C[2 * ldC + 2];
        c32 = C[2 * ldC + 3];
    }
    if (nr > 3) {
        c03 = C[3 * ldC + 0];
        c13 = C[3 * ldC + 1];
        c23 = C[3 * ldC + 2];
        c33 = C[3 * ldC + 3];
    }

    for (size_t p = 0; p < k; ++p) {
        const t1 a0 = Ap[p * MR + 0];
        const t1 a1 = Ap[p * MR + 1];
        const t1 a2 = Ap[p * MR + 2];
        const t1 a3 = Ap[p * MR + 3];

        const t2 b0 = Bp[p * NR + 0];
        const t2 b1 = Bp[p * NR + 1];
        const t2 b2 = Bp[p * NR + 2];
        const t2 b3 = Bp[p * NR + 3];

        c00 -= a0 * b0; c01 -= a0 * b1; c02 -= a0 * b2; c03 -= a0 * b3;
        c10 -= a1 * b0; c11 -= a1 * b1; c12 -= a1 * b2; c13 -= a1 * b3;
        c20 -= a2 * b0; c21 -= a2 * b1; c22 -= a2 * b2; c23 -= a2 * b3;
        c30 -= a3 * b0; c31 -= a3 * b1; c32 -= a3 * b2; c33 -= a3 * b3;
    }

    if (nr > 0) {
        const t3* Cj = C + 0 * ldC;
        if (mr > 0) Cj[0] = c00;
        if (mr > 1) Cj[1] = c10;
        if (mr > 2) Cj[2] = c20;
        if (mr > 3) Cj[3] = c30;
    }
    if (nr > 1) {
        const t3* Cj = C + 1 * ldC;
        if (mr > 0) Cj[0] = c01;
        if (mr > 1) Cj[1] = c11;
        if (mr > 2) Cj[2] = c21;
        if (mr > 3) Cj[3] = c31;
    }
    if (nr > 2) {
        const t3* Cj = C + 2 * ldC;
        if (mr > 0) Cj[0] = c02;
        if (mr > 1) Cj[1] = c12;
        if (mr > 2) Cj[2] = c22;
        if (mr > 3) Cj[3] = c32;
    }
    if (nr > 3) {
        const t3* Cj = C + 3 * ldC;
        if (mr > 0) Cj[0] = c03;
        if (mr > 1) Cj[1] = c13;
        if (mr > 2) Cj[2] = c23;
        if (mr > 3) Cj[3] = c33;
    }
}