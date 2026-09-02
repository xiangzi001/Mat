

template<typename T, typename t>
inline void __copy(T* dst, const t* src, int len) {
	for (int i = 0; i < len; ++i) {
		dst[i] = src[i];
	}
}
#ifdef RtlCopyMemory
template<typename T>
inline void __copy(T* dst, const T* src, int len) {
	RtlCopyMemory(dst, src, sizeof(T) * len);
}

#endif // RtlCopyMemory

#ifdef ZeroMemory
template<typename T>
inline void __set0(T* dst, int len) {
	ZeroMemory(dst, sizeof(T) * len);
}
#else
template<typename T>
inline void __set0(T* dst, int len) {
	for (int i = 0; i < len; ++i)
		dst[i] = T(0);
}
#endif // !ZeroMemory


#include "__Complex.h"

#include "__Matrix.h"
#include "__Vector.h"
#include "__Block.h"

#include "__Calculate.h"
#include "__times.h"