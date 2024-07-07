#include<iostream>
#include<stdio.h>
#include<omp.h>
#include<chrono>
#include<vector>

std::vector<int> NUM_THREADS_ARRAY{1,2,4,6,8,16,20,40};

//

void parallel_init_AC_and_B_vectors(std::vector<double> & vecA,
                                std::vector<double> & vecC,
                                std::vector<double> & vecB,
                                const int & m,
                                const int & n){
    vecB.reserve(n);
    //
    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        //  
        for (int i = lb; i <= ub; i++) {
            for (int j = 0; j < n; j++)
                vecA[i * n + j] = i + j;
            vecC[i] = 0.0;
        }     
    }
    for (int j = 0; j < n; j++) 
        vecB.push_back(j);
}


void matrix_vector_product_omp(std::vector<double> & vecA,
                                std::vector<double> & vecB,
                                std::vector<double> & vecC, 
                                const int & m,
                                const int & n,
                                const int & numThreads)
{
    #pragma omp parallel num_threads(numThreads)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        for (int i = lb; i <= ub; i++) {
            for (int j = 0; j < n; j++)
                vecC[i] += vecA[i * n + j] * vecB[j];
        }
    }
}


void general_programm(const int & M,
                      const int & N){
    //
    double T1;
    std::vector<double> vecA(M * N);
    std::vector<double> vecB;
    std::vector<double> vecC(M);

    parallel_init_AC_and_B_vectors(vecA, vecC, vecB, M, N);
    // 

    for (const auto num_thread: NUM_THREADS_ARRAY){

        std::vector<double> vecC_copy(vecC);

        const auto start{std::chrono::steady_clock::now()};
        //
        matrix_vector_product_omp(vecA, vecB, vecC_copy, M, N, num_thread);
        //
        const auto end{std::chrono::steady_clock::now()};
        auto elapsed_seconds = std::chrono::duration<double>(end - start); // seconds
        
        //
        if (num_thread == 1){
            T1 = elapsed_seconds.count();
            std::cout << "T1: " << T1 << '\n';
        }
        else {
            double T = elapsed_seconds.count();
            std::cout << "T" << num_thread << ": " << T << " S" << num_thread << ": " << T1/T << '\n';
        }
    }

    //    
}

int main(){
    //
    int M = 20000;
    int N = M;
    std::cout << ":" << M << '\n';
    //
    general_programm(M, N);
    //

    std::cout << '\n';

    
    M = 40000;
    N = M;
    std::cout << ":" << M << '\n'; 
    //
    general_programm(M, N);

    return 0;
}