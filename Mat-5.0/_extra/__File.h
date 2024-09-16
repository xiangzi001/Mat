
#include <fstream>

long int size(string name) {
	ifstream fr;
	fr.open(name);
	fr.seekg(0, ios::end);
	long int v = (int)fr.tellg();
	fr.close();
	return v;
}

template<class T>
void load(string name, T* data, int len) {
	ifstream fr;

	fr.open(name, ios_base::binary);
	fr.read((char*)(data), sizeof(T) * size);
	fr.close();
}
template<class T>
void load(string name, MAT<T>& A) {
	ifstream fr;

	fr.open(name, ios_base::binary);

	fr.read((char*)(&A.rows), 8);
	long s = A.rows * A.cols;
	delete[] A.data;
	A.data = new T[s];
	fr.read((char*)(A.data), sizeof(T) * s);

	fr.close();
}
template<class T>
void load(string name, MAT<T>& A, int r, int c) {
	ifstream fr;

	fr.open(name, ios_base::binary);
	A.init(r, c);
	fr.read((char*)(A.data), sizeof(T) * r * c);

	fr.close();
}



template<class T>
void save(string name, T* data, int len) {
	ofstream fw;

	fw.open(name, ios_base::binary);
	fw.write((char*)(data), sizeof(T) * len);
	fw.close();
}
template<class T>
void save(string name, MAT<T>& A) {
	ofstream fw;

	fw.open(name, ios_base::binary);

	fw.write((char*)(&A.rows), 8);
	fw.write((char*)(A.data), sizeof(T) * A.rows * A.cols);

	fw.close();
}
