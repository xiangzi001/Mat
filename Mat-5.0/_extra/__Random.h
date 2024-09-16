
template<class T>
void random(T& num, long long a = 25214903916) {
	//system_clock::time_point start = system_clock::now();
	//long long ms = (duration_cast<milliseconds>(start.time_since_epoch())).count();
	long long ms = (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();
	num = ((a + 1) * ms) % (25214903913);
}

template <typename T>
void random(MAT<T>& A, int r, int c, int Big, int Small) {
	A.rows = r;
	A.cols = c;
	int s = r * c;
	delete[] A.data;
	A.data = new T[s];
	int temp = r;
	for (int i = 0; i < s; ++i) {
		random(temp, i);
		temp = fabs(temp);
		A.data[i] = (temp % (Big - Small)) + Small;
	}
	//for (int i = 0; i < s; ++i) {
	//	srand((int)(time(0)) + i);
	//	A.data[i] = (rand() % (Big - Small)) + Small;
	//}
}

template <typename T>
void random(MAT<complex<T>>& A, int r, int c, int Big, int Small) {
	A.rows = r;
	A.cols = c;
	int s = r * c;
	delete[] A.data;
	A.data = new complex<T>[s];
	int temp = r;
	for (int i = 0; i < s; ++i) {
		random(temp, i);
		temp = fabs(temp);
		A.data[i].real = (temp % (Big - Small)) + Small;

		random(temp, A.data[i].real);
		temp = fabs(temp);
		A.data[i].imag = (temp % (Big - Small)) + Small;
	}
	//for (int i = 0; i < s; ++i) {
	//	srand((int)(time(0)) + i);
	//	A.data[i] = (rand() % (Big - Small)) + Small;
	//}
}
