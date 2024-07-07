#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

const std::vector<int> NUM_THREADS_ARRAY{1, 2, 4, 6, 8, 16, 20, 40};
const std::vector<int> MATRIX_SIZES{20000, 40000};

void init_matrix_and_res_vector(std::vector<double>& vecA,
                                std::vector<double>& vecC,
                                const int& start_row,
                                const int& end_row,
                                const int& n) {
    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < n; ++j) {
            vecA[i * n + j] = i + j;
        }
        vecC[i] = 0.0;
    }
}

void init_vector(std::vector<double>& vecB,
                 const int& start_id,
                 const int& end_id) {
    for (int i = start_id; i < end_id; ++i)
        vecB[i] = i;
}

void matrix_vector_product(const std::vector<double>& vecA,
                           const std::vector<double>& vecB,
                           std::vector<double>& vecC,
                           const int& start_row,
                           const int& end_row,
                           const int& n) {
    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < n; ++j) {
            vecC[i] += vecA[i * n + j] * vecB[j];
        }
    }
}

void general_programm(const int& M,
                      const int& N,
                      const int& max_threads) {
    double T1 = 0.0;
    std::vector<double> vecA(M * N);
    std::vector<double> vecB(N);
    std::vector<double> vecC(M);

    std::vector<std::thread> threads;

    // init vecA, vecC
    int rows_per_thread = M / max_threads;
    for (int threadid = 0; threadid < max_threads; ++threadid) {
        int start_row = threadid * rows_per_thread;
        int end_row = (threadid == max_threads - 1) ? M : start_row + rows_per_thread;
        threads.emplace_back(init_matrix_and_res_vector, std::ref(vecA), std::ref(vecC), start_row, end_row, N);
    }
    for (auto& thread : threads) {
         thread.join(); 
    }
    threads.clear();

    // init vecB
    int items_per_thread = N / max_threads;
    for (int threadid = 0; threadid < max_threads; ++threadid) {
        int start_id = threadid * items_per_thread;
        int end_id = (threadid == max_threads - 1) ? N : start_id + items_per_thread;
        threads.emplace_back(init_vector, std::ref(vecB), start_id, end_id);
    }
    for (auto& thread : threads) {
       thread.join(); 
    }
    threads.clear();

    // matrix_vector_product
    for (const auto num_threads : NUM_THREADS_ARRAY) {
        const auto start = std::chrono::steady_clock::now();

        rows_per_thread = M / num_threads;
        for (int threadid = 0; threadid < num_threads; ++threadid) {
            int start_row = threadid * rows_per_thread;
            int end_row = (threadid == num_threads - 1) ? M : start_row + rows_per_thread;
            threads.emplace_back(matrix_vector_product, std::cref(vecA), std::cref(vecB), std::ref(vecC), start_row, end_row, N);
        }
        for (auto& thread : threads) {
            thread.join();
        }
        const auto end = std::chrono::steady_clock::now();
        threads.clear();

        auto elapsed_seconds = std::chrono::duration<double>(end - start).count(); // seconds

        if (num_threads == 1) {
            T1 = elapsed_seconds;
            std::cout << "T1: " << T1 << '\n';
        } else {
            double T = elapsed_seconds;
            std::cout << "T" << num_threads << ": " << T << " S" << num_threads << ": " << T1 / T << '\n';
        }
    }
}

int main() {
    int max_threads = std::thread::hardware_concurrency();
    for (const auto matrix_size : MATRIX_SIZES) {
        general_programm(matrix_size, matrix_size, max_threads);
        std::cout << '\n';
    }
    return 0;
}

