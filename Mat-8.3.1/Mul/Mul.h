
#include"__micro_kernel_packed.h"
#include"__micro_kernel_unpacked.h"
#include"__micro_kernel_packed_double.h"
#include"__micro_kernel_unpacked_double.h"

#include"__pack.h"
#include"__pack_double.h"

template<typename t1, typename t2, typename t3>
void __mulAB_scalar(const MAT<t1>& A, const MAT<t2>& B, MAT<t3>& C)
{
    const int M = C.rows;
    const int N = C.cols;
    const int K = A.cols;

    const int ldA = A.rows;
    const int ldB = B.rows;
    const int ldC = C.rows;

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            t3 sum = t3(0);
            for (int k = 0; k < K; ++k) {
                sum += A.data[i + (size_t)k * ldA]
                    * B.data[k + (size_t)j * ldB];
            }
            C.data[i + (size_t)j * ldC] = sum;
        }
    }
}

template<typename t1, typename t2, typename t3>
void __mulAB_outer_vec(const MAT<t1>& A, const MAT<t2>& B, MAT<t3>& C)
{
    const int M = C.rows;
    const int N = C.cols;
    const int K = A.cols;

    const int ldA = A.rows;
    const int ldB = B.rows;
    const int ldC = C.rows;

    __set0(C.data, (size_t)M * N);

    for (int k = 0; k < K; ++k) {
        t1* Ak = A.data + (size_t)k * ldA;

        for (int j = 0; j < N; ++j) {
            const t2 b = B.data[k + (size_t)j * ldB];
            t3* Cj = C.data + (size_t)j * ldC;

            __mixed(Cj, Ak, b, Cj, M);
        }
    }
}

template<typename t1, typename t2, typename t3>
void __mulAB_vec(const MAT<t1>& A, const MAT<t2>& B, MAT<t3>& C)
{
    const int M = C.rows;
    const int N = C.cols;
    const int K = A.cols;

    const int ldA = A.rows;
    const int ldB = B.rows;
    const int ldC = C.rows;

    __set0(C.data, (size_t)M * N);

    for (int j = 0; j < N; ++j) {
        t3* Cj = C.data + (size_t)j * ldC;

        for (int k = 0; k < K; ++k) {
            const t2 b = B.data[k + (size_t)j * ldB];
            t1* Ak = A.data + (size_t)k * ldA;

            // Cj = Cj + Ak * b
            __mixed(Cj, Ak, b, Cj, M);
        }
    }
}

template<typename t1, typename t2, typename t3>
void __mulAB_unpacked(
    const MAT<t1>& A, const MAT<t2>& B, MAT<t3>& C)
{
    constexpr int MR = __micro_kernel_mr<t1, t2, t3>();
    constexpr int NR = __micro_kernel_nr<t1, t2, t3>();

    const int M = C.rows;
    const int N = C.cols;
    const int K = A.cols;

    const int ldA = A.rows;
    const int ldB = B.rows;
    const int ldC = C.rows;

    __set0(C.data, (size_t)M * N);

    for (int jp = 0; jp < N; jp += NR) {
        const int nn = (N - jp < NR) ? (N - jp) : NR;

        for (int ip = 0; ip < M; ip += MR) {
            const int mm = (M - ip < MR) ? (M - ip) : MR;

            const t1* Ap = A.data + ip;
            const t2* Bp = B.data + (size_t)jp * ldB;
            t3* Cp = C.data + (size_t)jp * ldC + ip;

            if (mm == MR && nn == NR) {
                __micro_kernel_mul_unpacked_full(
                    Ap, ldA, Bp, ldB, Cp, ldC, K);
            }
            else {
                __micro_kernel_mul_unpacked_edge(
                    Ap, ldA, Bp, ldB, Cp, ldC, K, mm, nn);
            }
        }
    }
}

template<typename t1, typename t2, typename t3>
void __mulAB_packed(
    const t1* __restrict A_pack, const t2* __restrict B_pack, t3* __restrict C,
    int ldC, int mc, int nc, int kc)
{
    constexpr int MR = __micro_kernel_mr<t1, t2, t3>();
    constexpr int NR = __micro_kernel_nr<t1, t2, t3>();

    const int mr_panels = (mc + MR - 1) / MR;
    const int nr_panels = (nc + NR - 1) / NR;

    for (int jp = 0; jp < nr_panels; ++jp) {
        const t2* Bp = B_pack + (size_t)jp * kc * NR;
        const int nn = (nc - jp * NR < NR) ? (nc - jp * NR) : NR;

        for (int ip = 0; ip < mr_panels; ++ip) {
            const t1* Ap = A_pack + (size_t)ip * kc * MR;
            const int mm = (mc - ip * MR < MR) ? (mc - ip * MR) : MR;

            t3* Cp = C + (size_t)(jp * NR) * ldC + ip * MR;

            if (mm == MR && nn == NR) {
                __micro_kernel_mul_packed_full(Ap, Bp, Cp, ldC, kc);
            }
            else {
                __micro_kernel_mul_packed_edge(Ap, Bp, Cp, ldC, kc, mm, nn);
            }
        }
    }
}

template<typename t1, typename t2, typename t3>
void __mulAB_2(const MAT<t1>& A, const MAT<t2>& B, MAT<t3>& C) {
    constexpr int MR = __micro_kernel_mr<t1, t2, t3>();
    constexpr int NR = __micro_kernel_nr<t1, t2, t3>();

    constexpr int MC = __BlockSize_Mul_MC;
    constexpr int KC = __BlockSize_Mul_KC;
    constexpr int NC = __BlockSize_Mul_NC;

    const int M = C.rows;
    const int N = C.cols;
    const int K = A.cols;

    __set0(C.data, (size_t)M * N);

#define __align_up(x, a) (x + a - 1) & ~(a - 1)
    const size_t A_pack_size = __align_up(sizeof(t1) * ((MC + MR - 1) / MR) * KC * MR, 64);
    const size_t B_pack_size = __align_up(sizeof(t2) * ((NC + NR - 1) / NR) * KC * NR, 64);
#undef __align_up

    char* pack_buf = (char*)_mm_malloc(A_pack_size + B_pack_size, 64);
    t1* A_pack = (t1*)(pack_buf);
    t2* B_pack = (t2*)(pack_buf + A_pack_size);

    for (int pc = 0; pc < K; pc += KC) {
        const int kc = (K - pc < KC) ? (K - pc) : KC;

        for (int ic = 0; ic < M; ic += MC) {
            const int mc = (M - ic < MC) ? (M - ic) : MC;

            __pack_A_block<t1, t2>(A.data + (size_t)pc * A.rows + ic,
                A.rows, mc, kc, A_pack);

            for (int jc = 0; jc < N; jc += NC) {
                const int nc = (N - jc < NC) ? (N - jc) : NC;

                __pack_B_block<t1, t2>(B.data + (size_t)pc + (size_t)jc * B.rows,
                    B.rows, kc, nc, B_pack);

                __mulAB_packed<t1, t2, t3>(
                    A_pack, B_pack,
                    C.data + (size_t)jc * C.rows + ic,
                    C.rows, mc, nc, kc);
            }
        }
    }

    _mm_free(pack_buf);
}

template<typename t1, typename t2>
auto operator * (const MAT<t1>& A, const MAT<t2>& B) {
    MAT<decltype(std::declval<t1>()* std::declval<t2>())> C(A.rows, B.cols);

    const int M = C.rows;
    const int N = C.cols;
    const int K = A.cols;

    if ((M <= 4 && N <= 4 && K <= 4) || (long long)M * (long long)N * (long long)K <= 64) {
        __mulAB_scalar(A, B, C);
        return C;
    }

    if (K <= 4) {
        __mulAB_outer_vec(A, B, C);
        return C;
    }

    if ((M <= 32 && N <= 32 && K <= 32) || K <= 16) {
        __mulAB_vec(A, B, C);
        return C;
    }

    if (M <= 128 && N <= 128 && K <= 128) {
        __mulAB_unpacked(A, B, C);
        return C;
    }

    __mulAB_2(A, B, C);

    return C;
}



template<typename t>
static void __mulAB_packed_n(
    const t* __restrict A_pack,
    const t* __restrict B_pack,
    t* __restrict C, size_t ldC,
    int m, int n, int k)
{
    constexpr int MR = __micro_kernel_mr<t, t, t>();
    constexpr int NR = __micro_kernel_nr<t, t, t>();
    const int mp = (m + MR - 1) / MR;
    const int np = (n + NR - 1) / NR;
    for (int jp = 0; jp < np; ++jp) {
        const t* Bp = B_pack + (size_t)jp * k * NR;
        const int nn = (n - jp * NR < NR) ? (n - jp * NR) : NR;
        for (int ip = 0; ip < mp; ++ip) {
            const t* Ap = A_pack + (size_t)ip * k * MR;
            const int mm = (m - ip * MR < MR) ? (m - ip * MR) : MR;
            t* Cp = C + (size_t)(jp * NR) * ldC + ip * MR;
            if (mm == MR && nn == NR)
                __micro_kernel_nmul_packed_full(Ap, Bp, Cp, ldC, (size_t)k);
            else
                __micro_kernel_nmul_packed_edge(Ap, Bp, Cp, ldC, (size_t)k, mm, nn);
        }
    }
}

template<typename t>
static void __mulAB_lower_tri_packed_n(
    const t* __restrict A_pack,
    const t* __restrict B_pack,
    t* __restrict C, size_t ldC,
    int m, int n, int k, int k_base)
{
    constexpr int MR = __micro_kernel_mr<t, t, t>();
    constexpr int NR = __micro_kernel_nr<t, t, t>();
    const int mp = (m + MR - 1) / MR;
    const int np = (n + NR - 1) / NR;
    for (int jp = 0; jp < np; ++jp) {
        const t* Bp_base = B_pack + (size_t)jp * k * NR;
        const int nn = (n - jp * NR < NR) ? (n - jp * NR) : NR;
        for (int ip = 0; ip < mp; ++ip) {
            const t* Ap = A_pack + (size_t)ip * k * MR;
            const int mm = (m - ip * MR < MR) ? (m - ip * MR) : MR;
            const int k_eff = std::min(k, k_base + ip * MR + mm);
            if (k_eff <= 0) continue;
            t* Cp = C + (size_t)(jp * NR) * ldC + ip * MR;
            if (mm == MR && nn == NR)
                __micro_kernel_nmul_packed_full(Ap, Bp_base, Cp, ldC, (size_t)k_eff);
            else
                __micro_kernel_nmul_packed_edge(Ap, Bp_base, Cp, ldC, (size_t)k_eff, mm, nn);
        }
    }
}

template<typename t>
static void __mulAB_upper_tri_packed_n(
    const t* __restrict A_pack,
    const t* __restrict B_pack,
    t* __restrict C, size_t ldC,
    int m, int n, int k, int k_base)
{
    constexpr int MR = __micro_kernel_mr<t, t, t>();
    constexpr int NR = __micro_kernel_nr<t, t, t>();
    const int mp = (m + MR - 1) / MR;
    const int np = (n + NR - 1) / NR;
    for (int jp = 0; jp < np; ++jp) {
        const t* Bp_base = B_pack + (size_t)jp * k * NR;
        const int nn = (n - jp * NR < NR) ? (n - jp * NR) : NR;
        for (int ip = 0; ip < mp; ++ip) {
            const t* Ap_base = A_pack + (size_t)ip * k * MR;
            const int mm = (m - ip * MR < MR) ? (m - ip * MR) : MR;
            const int k_start = std::max(0, ip * MR - k_base);
            const int k_eff = k - k_start;
            if (k_eff <= 0) continue;
            const t* Ap = Ap_base + (size_t)k_start * MR;
            const t* Bp = Bp_base + (size_t)k_start * NR;
            t* Cp = C + (size_t)(jp * NR) * ldC + ip * MR;
            if (mm == MR && nn == NR)
                __micro_kernel_nmul_packed_full(Ap, Bp, Cp, ldC, (size_t)k_eff);
            else
                __micro_kernel_nmul_packed_edge(Ap, Bp, Cp, ldC, (size_t)k_eff, mm, nn);
        }
    }
}