// FockStateSimulation.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <vector>
#include <complex>
#include <iostream>
#include <fstream>
#include <cmath>
#include <math.h>

using namespace std;

complex <long double> ni(0, -1);
complex<long double> half(0.5, 0);
complex<long double> six(6, 0);
complex<long double> twice(2, 0);

void print_complex(vector<complex<long double>> to_print, size_t end) {
	for (size_t i = 0; i < end; i++) {
		cout << to_print[i] << " ";
	}
	cout << "\n";
}

void print_double(vector<long double> to_print, size_t end) {
	for (size_t i = 0; i < end; i++) {
		cout << to_print[i] << " ";
	}
	cout << "\n";
}

void tri_ham(long double c, long double bfield, long double n_atoms, vector<complex<long double>> &psi, vector<complex<long double>> &ans) {
	for (size_t i = 0; i < ans.size(); i++) {
		ans[i] = (i*(2 * (n_atoms - 2 * i)) - 1)* c / n_atoms*psi[i] + 2 * bfield * i*psi[i];
	}
	for (size_t i = 1; i < ans.size(); i++) {
		ans[i] += i * sqrt((n_atoms - 2 * (i - 1) - 1)*(n_atoms - 2 * (i - 1)))*psi[i - 1] * c / n_atoms;
	}
	for (size_t i = 0; i < ans.size() - 1; i++) {
		ans[i] += (i + 1)*sqrt((n_atoms - 2 * (i + 1) + 1)*(n_atoms - 2 * (i + 1) + 2))*psi[i + 1] * c / n_atoms;
	}
}

void func_to_integrate(vector<complex<long double>> &ans,vector<complex<long double>> &yn, complex<long double> t, long double bfield, long double c, long double n_atoms) {
	tri_ham(c, bfield, n_atoms, yn, ans);
	for (size_t i = 0; i < ans.size(); i++) {
		ans[i] = ans[i] * ni;
	}
}


void ynplus1(vector<complex<long double>> &yn, vector<complex<long double>> &k1, vector<complex<long double>> &k2, vector<complex<long double>> &k3, vector<complex<long double>> &k4, vector<complex<long double>> &ktemp,
					complex<long double> t , complex<long double> dt , long double bfield, long double c, long double n_atoms) {
	func_to_integrate(k1, yn, t, bfield, c, n_atoms);
	for (size_t i = 0; i < ktemp.size(); i++) {
		ktemp[i] = yn[i] + dt*half*k1[i];
	}
	func_to_integrate(k2, ktemp, t+dt*half, bfield, c, n_atoms);
	for (size_t i = 0; i < ktemp.size(); i++) {
		ktemp[i] = yn[i] + dt*half*k2[i];
	}
	func_to_integrate(k3, ktemp, t+dt*half, bfield, c, n_atoms);
	for (size_t i = 0; i < ktemp.size(); i++) {
		ktemp[i] = yn[i] + dt*k3[i];
	}
	func_to_integrate(k4, ktemp, t+dt, bfield, c, n_atoms);

	for (size_t i = 0; i < ktemp.size(); i++) {
		yn[i] = yn[i] + dt*(k1[i]+k2[i]*twice+k3[i]*twice+k4[i])/six;
	}
	
}

//use nostd to keep track of nosqr
void calc_n0_vals(long double &n0, long double &n0std, vector<complex<long double>> &psi, long double num_atoms) {
	n0 = 0;
	n0std = 0;
	long double n0var;
	for (size_t k = 0; k < psi.size(); k++) {
		n0 += (num_atoms - 2 * k) * norm(psi[k]);
		n0std += pow(num_atoms - 2 * k,2) * norm(psi[k]);
	}
	n0std = sqrt(n0std - pow(n0, 2));
}
int main()
{
	long double c = 30;
	long double bfield= -30.165300000000002;
	long double n_atoms = 10000;
	complex<long double> dt(1e-6, 0);
	vector<complex<long double>> psi;
	psi.resize(int(n_atoms/2) + 1);
	//initialize variables we will need 
	vector<complex<long double>> ktemp;
	vector<complex<long double>> k1;
	vector<complex<long double>> k2;
	vector<complex<long double>> k3;
	vector<complex<long double>> k4;
	ktemp.resize(psi.size());
	k1.resize(psi.size());
	k2.resize(psi.size());
	k3.resize(psi.size());
	k4.resize(psi.size());
	//initialize state
	psi[0] = 1;
	//initalize answer vectors
	vector<long double> t;
	vector<long double> n0;
	vector<long double> n0std;
	long double n0_in;
	long double n0std_in;
	//do computation
	cout.precision(18);
	for (int i=0; i < 100000; i++) {
		complex<long double> ii(i, 0);
		calc_n0_vals(n0_in, n0std_in, psi, n_atoms);
		t.push_back(real(ii*dt));
		n0.push_back(n0_in);
		n0std.push_back(n0std_in);
		ynplus1(psi, k1, k2, k3, k4, ktemp, ii*dt, dt, bfield, c, n_atoms);
	}
	
	ofstream myfile;
	myfile.open("C:\\Users\\Administrator\\Documents\\TESTING\\cpptesting.txt", ios::out);
	myfile.precision(17);
	for (size_t i = 0; i < n0.size(); i++) {
		myfile << t[i] << "  " << n0[i] << "  " << n0std[i] << "\n";
	}
	myfile.close();
    return 0;
}

