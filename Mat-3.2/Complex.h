
template<typename T>
class complex {
public:
	T real;
	T imag;

	complex() :
		real(0), imag(0)
	{
	}
	template<typename t>
	complex(t a) :
		real(a), imag(0)
	{
	}
	template<typename t_1, typename t_2>
	complex(t_1 r, t_2 i) :
		real(r), imag(i)
	{
	}

	void operator = (T a) {
		real = a;
		imag = 0;
	}

	complex operator - () {
		return { -real , -imag };
	}

	complex operator + (T b) {
		return { real + b , imag };
	}
	complex operator - (T b) {
		return { real - b , imag };
	}
	complex operator / (T b) {
		return { real / b , imag / b };
	}

	complex operator * (T b) {
		return { real * b , imag * b };
	}

	template<typename t>
	complex operator + (complex<t> b) {
		return { real + b.real, imag + b.imag };
	}
	template<typename t>
	complex operator - (complex<t> b) {
		return { real - b.real, imag - b.imag };
	}
	template<typename t>
	complex operator * (complex<t> b) {
		return { real * b.real - imag * b.imag , real * b.imag + imag * b.real };
	}
	template<typename t>
	complex operator / (complex<t> b) {
		T temp(1 / (b.real * b.real + b.imag * b.imag));
		return { (b.real * real + b.imag * imag) * temp, (b.real * imag - b.imag * real) * temp };
	}

	void operator += (T b) {
		real = real + b;
	}
	void operator -= (T b) {
		real = real - b;
	}
	void operator /= (T b) {
		real = real / b;
		imag = imag / b;
	}
	void operator *= (T b) {
		real = real * b;
		imag = imag * b;
	}

	template<typename t>
	void operator += (complex<t> b) {
		real = real + b.real;
		imag = imag + b.imag;
	}
	template<typename t>
	void operator -= (complex<t> b) {
		real = real - b.real;
		imag = imag - b.imag;
	}
	template<typename t>
	void operator *= (complex<t> b) {
		T r(real * b.real - imag * b.imag);
		imag = real * b.imag + imag * b.real;
		real = r;
	}
	template<typename t>
	void operator /= (complex<t> b) {
		T temp(1 / (b.real * b.real + b.imag * b.imag));
		T r((b.real * real + b.imag * imag)* temp);
		imag = (b.real * imag - b.imag * real) * temp;
		real = r;
	}

	template<typename t>
	bool operator == (complex<t> b) {
		return (real == b.real) and (imag == b.imag);
	}
	template<typename t>
	bool operator == (t b) {
		return (real == b) and (imag == 0);
	}
	bool is_zero() {
		return (real == 0) and (imag == 0);
	}

	template<typename t>
	bool operator != (complex<t> b) {
		return (real != b.real) or (imag != b.imag);
	}
	template<typename t>
	bool operator != (t b) {
		return (real != b) or (imag != 0);
	}

	complex conj() {
		return { real , -imag };
	}

	T len() {
		return sqrt(real * real + imag * imag);
	}

	T len2() {
		return real * real + imag * imag;
	}

	T fabs() {
		return fabs(real) + fabs(imag);
	}

	complex sqrt() {
		T c(sqrt(real * real + imag * imag));
		return { sqrt(0.5 * (c + real)), sign(imag) * sqrt(0.5 * (c - real)) };
	}

	complex pow(double n) {
		T r(pow(real * real + imag * imag, 0.5 * n));
		double theta(atan2(imag, real) * n);
		return { r * cos(theta) , r * sin(theta) };
	}
};

template<typename t, typename T>
complex<T> operator / (t a, complex <T> b) {
	T temp = a / (b.real * b.real + b.imag * b.imag);
	return { b.real * temp, - b.imag * temp };
}

template<typename t, typename T>
complex<T> operator * (t a, complex <T> b) {
	return { b.real * a, b.imag * a };
}

template<typename t, typename T>
complex<T> operator + (t a, complex <T> b) {
	return { b.real + a, b.imag };
}

template<typename T>
char sign(T x) {
	if (x > 0) {
		return 1;
	}
	else {
		return -1;
	}
}

template<typename T>
complex<T> sign(complex<T> X) {
	return(X / sqrt(X.real * X.real + X.imag * X.imag));
}

template<typename T>
complex<T> conj(complex<T> X) {
	return { X.real , -X.imag };
}

template<typename T>
T len(complex<T> X) {
	return sqrt(X.real * X.real + X.imag * X.imag);
}
template<typename T>
T len2(complex<T> X) {
	return X.real * X.real + X.imag * X.imag;
}

template<typename T>
T fabs(complex<T> X) {
	return fabs(X.real) + fabs(X.imag);
}

template<typename T>
complex<T> sqrt(complex<T> X) {
	//T c(sqrt(X.real * X.real + X.imag * X.imag));
	//return { sqrt(0.5 * (c + X.real)), sign(X.imag) * sqrt(0.5 * (c - X.real)) };
	T r(pow(X.real * X.real + X.imag * X.imag, 0.25));
	double theta(atan2(X.imag, X.real) * 0.5);
	return { r * cos(theta) , r * sin(theta) };
}

template<typename T>
complex<T> pow(complex<T> X, double n) {
	T r(pow(X.real * X.real + X.imag * X.imag, 0.5 * n));
	double theta(atan2(X.imag, X.real) * n);
	return { r * cos(theta) , r * sin(theta) };
}

template<typename T, typename t>
void solve_quadratic(complex<T>& i, complex<T>& j, t a, t b, t c) {
	T d(b * b - 4 * a * c);
	if (d < 0) {
		d = sqrt(-d);
		T temp(0.5 / a);
		i.real = -b * temp;
		i.imag = d * temp;
		j.real = -b * temp;
		j.imag = -d * temp;
	}
	else {
		d = sqrt(d);
		T temp(0.5 / a);
		i.real = (-b + d) * temp;
		i.imag = 0;
		j.real = (-b - d) * temp;
		j.imag = 0;
	}
}

template<typename T>
complex<T> times_conj (complex<T>& a, complex<T>& b) {
	return { a.real * b.real + a.imag * b.imag , -a.real * b.imag + a.imag * b.real };
}
