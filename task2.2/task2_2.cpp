#include<iostream>
#include<stdio.h>
#include<omp.h>
#include<chrono>
#include<vector>
#include<cmath>

const std::vector<int> NUM_TREADS_ARRAY{1,2,4,6,8,16,20,40};
const std::vector<int> nsteps_array{40000000, 80000000};
const double a = -4;
const double b = 4;

//

double func(const double & x) {
	return std::exp(x * -x);
}

double integrate_omp(double (*func)(const double &),
			const double & a,
			const double & b,
			const int & n,
			const int & numThreads)
{
	double h = (b - a) / n;
	double sum = 0.0;
	
	#pragma omp parallel num_threads(numThreads)
	{
		int nthreads = omp_get_num_threads();
		int threadid = omp_get_thread_num();
		int items_per_thread = n / nthreads;
		int lb = threadid * items_per_thread;
		int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);
		double sumloc = 0.0;
		
		for (int i = lb; i <= ub; i++)
			sumloc += func(a + h * (i + 0.5));
		
		#pragma omp atomic
		sum += sumloc;
	}
	sum *= h;
	return sum;
}



int main() {
	//
	double T1;

	for (auto nsteps : nsteps_array) {
		for (const auto num_threads: NUM_TREADS_ARRAY) {
			
			const auto start{std::chrono::steady_clock::now()};		
			//
			std:: cout << "sum: " << integrate_omp(func, a, b, nsteps, num_threads) << ' ';	
			//	
			const auto end{std::chrono::steady_clock::now()};
			auto elapsed_seconds = std::chrono::duration<double>(end - start); // seconds	
			
			//
			if (num_threads == 1){
				T1 = elapsed_seconds.count();
				std::cout << "T1: " << T1 << '\n';
			}
			else {
				double T = elapsed_seconds.count();
				std::cout << "T" << num_threads << ": " << T << " S" << num_threads << ": " << T1/T << '\n';		
			}
		}
		
		std::cout << '\n';
	}
	return 0;
}


