#include <iostream>
#include <vector>
#include <time.h>
#include <set>
#include <chrono>
#include <random>

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

int task2(){

}

int main(){
    task1();

    return 0;
}