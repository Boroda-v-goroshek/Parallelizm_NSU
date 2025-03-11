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

double ex3_1(const int N, const double eps, double t, int threads_num) {
    omp_set_num_threads(threads_num);

    std::vector<double> x(N);
    std::vector<double> Ax(N);
    std::vector<double> b(N);

    double time = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        b[i] = N + 1; 
    }

    while (true) {
        double max_diff = 0.0;

        // Параллельный цикл для вычисления Ax
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            Ax[i] = 0.0; // Сброс значения Ax
            for (int j = 0; j < N; ++j) {
                Ax[i] += (i == j ? 2.0 : 1.0) * x[j]; // A[i][j] = 2 если i==j, иначе 1
            }
            Ax[i] = Ax[i] - b[i];
        }

        // Параллельный цикл для обновления x и поиска max_diff
        #pragma omp parallel for reduction(max:max_diff)
        for (int i = 0; i < N; ++i) {
            double new_value = x[i] - t * Ax[i];
            max_diff = std::max(max_diff, std::abs(new_value - x[i]));
            x[i] = new_value;
        }

        if (max_diff < eps) {
            break;
        }
    }

    time = omp_get_wtime() - time;
    return time;
}

double ex3_2(const int N, const double eps, double t, int threads_num) {
    omp_set_num_threads(threads_num);

    std::vector<double> x(N);
    std::vector<double> Ax(N);
    std::vector<double> b(N);

    double time = omp_get_wtime();
    #pragma omp parallel
    {

        #pragma omp for
        for (int i = 0; i < N; ++i) {
            b[i] = N + 2;
        }

        while (true) {
            double max_diff = 0.0;

            // Параллельный цикл для вычисления Ax
            #pragma omp for
            for (int i = 0; i < N; ++i) {
                Ax[i] = 0.0; 
                for (int j = 0; j < N; ++j) {
                    Ax[i] += (i == j ? 2.0 : 1.0) * x[j]; // A[i][j] = 2 если i==j, иначе 1
                }
                Ax[i] = Ax[i] - b[i]; // A*x_n - b
            }

            double local_max_diff = 0.0;
            // Параллельный цикл для обновления x и поиска max_diff
            #pragma omp for
            for (int i = 0; i < N; ++i) {
                double new_value = x[i] - t * Ax[i];
                local_max_diff = std::max(local_max_diff, std::abs(new_value - x[i])); // Обновление локального max_diff
                x[i] = new_value;
            }

            #pragma omp critical
            {
                max_diff = std::max(max_diff, local_max_diff);
            }
            if (max_diff < eps) {
                break;
            }
        }
    }

    time = omp_get_wtime() - time;
    return time;
}


int task3(){

    int N = 50000;
    double t = 0.1;
    double eps = 1e-5;
    double sequential_time1, sequential_time2;

    set<int> threads = {1, 2, 4, 8, 16, 20, 40};
    for (auto threads_num : threads){
        double time1 = ex3_1(N, t, eps, threads_num);
        double time2 = ex3_2(N, t, eps, threads_num);

        if (threads_num == 1){
            sequential_time1 = time1;
            sequential_time2 = time2;
        }

        double accelerate1 = sequential_time1 / time1;
        double accelerate2 = sequential_time2 / time2;

        cout << threads_num << " : " << accelerate1 << " " << accelerate2 << endl;
    }
    return 0;
}

int main(){
    task3();

    return 0;
}