

template<typename t>
ostream& operator << (ostream& out, complex<t> s)
{
	out << s.real;
	if (s.imag < 0) {
		out << "-" << -s.imag << "i";
	}
	else {
		out << "+" << s.imag << "i";
	}

	return out;
}

template<typename T>
ostream& operator << (ostream& out, Func<T>& f) {
	for (int x = f.times; x > 0; --x) {
		out << f.data[x] << "*x^" << x << "+ ";
	}
	out << f.data[0] << endl;

	return out;
}
template<typename T>
ostream& operator << (ostream& out, FuncData<T> f) {
	for (int x = f.times; x > 0; --x) {
		out << f.data[x] << " x^" << x << "+	";
	}
	out << f.data[0] << endl;

	delete[] f.data;
	return out;
}

template<typename T>
ostream& operator << (ostream& out, MAT<T>& A) {
	for (int x = 0; x < A.rows; ++x) {
		for (int y = 0; y < A.cols; ++y) {
			out << A[x][y] << ",	";
		}
		out << endl;
	}
	out << endl;

	return out;
}
template<typename T>
ostream& operator << (ostream& out, MAT<T> A) {
	for (int x = 0; x < A.rows; ++x) {
		for (int y = 0; y < A.cols; ++y) {
			out << A[x][y] << ",	";
		}
		out << endl;
	}
	out << endl;

	delete[] A.data;
	return out;
}

template<class T>
void print(MAT<T>& A) {
	for (int x = 0; x < A.rows; x++) {
		for (int y = 0; y < A.cols; y++) {
			cout << A[x][y] << "," << "	";
		}
		cout << endl;
	}
	cout << endl;
}
template<class T>
void print(MAT<T>&& A) {
	for (int x = 0; x < A.rows; x++) {
		for (int y = 0; y < A.cols; y++) {
			cout << A[x][y] << "," << "	";
		}
		cout << endl;
	}
	cout << endl;
}
