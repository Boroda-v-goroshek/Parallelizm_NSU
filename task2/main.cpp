#include <iostream>
#include <vector>
#include <time.h>
#include <set>
#include <chrono>
#include <random>
#include <cmath>

#include <omp.h>

using namespace std;
int task1(){

    set<int> threads = {1,2,4,7,8,16,20,40};
    set<int> size = {20000, 40000};

    vector<vector<double>> result_times = vector<vector<double>> (size.size());
    vector<vector<double>> result_accelerations = vector<vector<double>> (size.size());
    for (int i = 0; i < size.size(); i++){
        result_times[i] = vector<double> (threads.size());
        result_accelerations[i] = vector<double> (threads.size());
    }

    double sequentiall_time;
    int str_id = 0;

    for (auto cur_size : size){
        
        int col_id = 0;
        for (auto threads_num : threads){                
            omp_set_num_threads(threads_num);

            vector<vector<double>> matrix = vector<vector<double>> (cur_size);
            vector<double> vec = vector<double> (cur_size);
            vector <double> result = vector<double> (cur_size);

            auto start = omp_get_wtime();

            
            for (int i = 0; i < cur_size; i++){
                matrix[i] = vector<double> (cur_size);
            }
            mt19937 generator(omp_get_thread_num());
            uniform_real_distribution<double> distribution(0.0, 1.0);
            for (int i = 0; i < cur_size; i++){
                vec[i] = distribution(generator);
                result[i] = 0;
            }

            #pragma omp parallel
            {
                // Генератор случайных чисел для каждого потока
                mt19937 generator(omp_get_thread_num()); // Сид с использованием номера потока
                uniform_real_distribution<double> distribution(0.0, 1.0);

                #pragma omp for
                for (int i = 0; i < cur_size; i++){
                    for(int j = 0; j < cur_size; j++){
                        matrix[i][j] = distribution(generator); // Заполнение значением
                    }
                }
            
                std::vector<double> local_result(cur_size, 0.0);

                #pragma omp for
                for (int i = 0; i < cur_size; i++) {
                    for (int j = 0; j < cur_size; j++) {
                        result[i] += matrix[i][j] * vec[j];
                    }
                }

                
            }

            auto end = omp_get_wtime();
            auto time = end - start;

            result_times[str_id][col_id] = time;

            if (threads_num == 1)
                sequentiall_time = time;
            result_accelerations[str_id][col_id] = sequentiall_time / time;
            
            col_id++;
            cout << "Стадия: " << cur_size << ", " << threads_num << endl;
        }
        str_id++;
    }

    for (int i = 0; i < size.size(); i++){
        for (int j = 0; j < threads.size(); j++){
            cout << result_times[i][j] << ' ';
        }
        cout << endl;
    }

    cout << "----------"<< endl;

    for (int i = 0; i < size.size(); i++){
        for (int j = 0; j < threads.size(); j++){
            cout << result_accelerations[i][j] << ' ';
        }
        cout << endl;
    }

    return 0;
}

double f(double x){
    return x * x * x * x * x;
}

int task2(){
    set<int> threads = {1,2,4,7,8,16,20,40};
    int steps = 200000000;

    vector<double> result_times = vector<double> (threads.size());
    vector<double> result_accelerations = vector<double> (threads.size());

    int id = 0;

    for (auto threads_num : threads){
        double sequentiall_time;

        omp_set_num_threads(threads_num);

        double a = 1.0;
        double b = 4.0;
        double h = (b - a) / steps;

        double sum = 0.0;

        auto start = omp_get_wtime();
        #pragma omp parallel
        {
            double local_sum = 0.0;
            
            #pragma omp for
            for (int i = 0; i < steps; i++){
                double x = a + i * h;
                local_sum += f(x);
            }

            #pragma omp atomic
            sum += local_sum * h;

        }
        auto end = omp_get_wtime();
        auto time = end - start;

        result_times[id] = time;
        if (threads_num == 1)
            sequentiall_time = time;
        result_accelerations[id] = sequentiall_time / time;


        id++;
        cout << "Стадия: " << threads_num << endl;
    }

    for (int j = 0; j < threads.size(); j++){
        cout << result_times[j] << ' ';
    }
    cout << endl;

    cout << "----------"<< endl;

    for (int j = 0; j < threads.size(); j++){
        cout << result_accelerations[j] << ' ';
    }
    cout << endl;
    

    return 0;
}

double ex3_1(vector<vector<double>> A, vector<double> x0, double t = 0.01, vector<double> b, int threads_num){
    omp_set_num_threads(threads_num);

    int N = x0.size();
    double eps = 0.01, mod = 100000.0;
    vector<double> x = x0;
    vector<double> x_new = x0;

    auto start = omp_get_wtime();
    while(mod > eps) {
        vector<double> Ax(N);
        double local_sum = 0.0;

        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            double temp_sum = 0.0; // Временная сумма для текущего потока
            for (int j = 0; j < N; ++j) {
                temp_sum += A[i][j] * x[j]; 
            }
            Ax[i] = temp_sum + b[i]; 
        }

        // Теперь собираем суммарный модуль из локальных значений
        #pragma omp for parallel reduction(+:local_sum)
        for (int i = 0; i < N; ++i) {
            local_sum += pow(Ax[i], 2);
        }

        mod = sqrt(local_sum);

        vector<double> tAxPlusB(N);
        #pragma omp parallel for
        for (size_t i = 0; i < N; ++i) {
            tAxPlusB[i] = x[i] * t; 
        }
        
        #pragma omp parallel for
        for (size_t i = 0; i < N; ++i) {
            x_new[i] = x[i] - tAxPlusB[i]; 
        }

        x = x_new;
    }
    auto end = omp_get_wtime();
    return end - start;
}

double ex3_2(vector<vector<double>> A, vector<double> x0, double t = 0.01, vector<double> b, int threads_num){
    omp_set_num_threads(threads_num);

    int N = x0.size();
    double eps = 0.01, mod = 100000.0;
    vector<double> x = x0;
    vector<double> x_new = x0;

    auto start = omp_get_wtime();

    #pragma omp parallel
    while(mod > eps) {
        vector<double> Ax(N); // Локальный вектор Ax для каждого потока
        double local_sum = 0.0; // Локальная сумма для каждого потока

        #pragma omp for
        for (int i = 0; i < N; ++i) {
            double temp_sum = 0.0; // Временная сумма для текущего потока
            for (int j = 0; j < N; ++j) {
                temp_sum += A[i][j] * x[j]; 
            }
            Ax[i] = temp_sum + b[i]; 
        }

        // Теперь собираем суммарный модуль из локальных значений
        #pragma omp for reduction(+:local_sum)
        for (int i = 0; i < N; ++i) {
            local_sum += pow(Ax[i], 2);
        }

        // Синхронизируем
        #pragma omp single
        {
            mod = sqrt(local_sum);
        }

        vector<double> tAxPlusB(N);
        #pragma omp for
        for (size_t i = 0; i < N; ++i) {
            tAxPlusB[i] = x[i] * t; 
        }
        
        #pragma omp for
        for (size_t i = 0; i < N; ++i) {
            x_new[i] = x[i] - tAxPlusB[i]; 
        }

        // Синхронизируем
        #pragma omp single
        {
            x = x_new; 
        }
    }
    auto end = omp_get_wtime();
    return end - start;
}

int task3(){

    int N = 3000;
    vector<double> x0 = vector<double>(N);
    vector<vector<double>> A = vector<vector<double>>(N);
    for (int i = 0; i < N; i++){
        A[i] = vector<double>(N);
    }
    vector<double> b = vector<double>(N);

    // Инициализация
    #pragma omp parallel
    {    
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<double> distribution(-1.0, 1.0);
        #pragma omp for
        for (int i = 0; i < N; ++i){
            x0[i] = distribution(gen);
            b[i] = N + 1;
            for (int j = 0; j < N; j++){
                A[i][j] = 1;
                if (i == j)
                    A[i][j]++;
            }
        }
        
    }
    double t = 0.01;
    double sequential_time1, sequential_time2;

    set<int> threads = {1, 2, 4, 8, 16, 20, 40};
    for (auto threads_num : threads){
        double time1 = ex3_1(A, x0, t, b, threads_num);
        double time2 = ex3_2(A, x0, t, b, threads_num);

        if (threads_num == 1){
            sequential_time1 = time1;
            sequential_time2 = time2;
        }

        double accelerate1 = sequential_time1 / time1;
        double accelerate2 = sequential_time2 / time2;

        cout << threads_num << " : " << accelerate1 << accelerate2 << endl;
    }
    return 0;
}

int main(){
    task3();

    return 0;
}