#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

int ex3_2() {
    const int N = 1000; // Выберите подходящее значение N
    const double eps = 1e-5;
    double t = 1.0;     // Начальное значение для t
    std::vector<double> x(N, 0.0);
    std::vector<double> Ax(N, 0.0);
    std::vector<double> b(N);

    // Инициализация вектора b
    for (int i = 0; i < N; ++i) {
        b[i] = i + 2;   // Значения от 2 до N + 1
    }

    double time = omp_get_wtime(); // Начало измерения времени

    #pragma omp parallel
    {
        while (true) {
            double max_diff = 0.0;

            // Параллельный цикл для вычисления Ax
            #pragma omp for
            for (int i = 0; i < N; ++i) {
                Ax[i] = 0.0; // Сброс значения Ax
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
                x[i] = new_value; // Обновление x
            }

            // Объединение max_diff из всех потоков
            #pragma omp critical
            {
                max_diff = std::max(max_diff, local_max_diff);
            }

            if (max_diff < eps) {
                break; // Условие выхода, если достигнуто требуемое приближение
            }
        }
    }

    time = omp_get_wtime() - time; // Завершение измерения времени
    std::cout << "Время выполнения (вариант 2, N = " << N << "): " << time << " секунд." << std::endl;

    return 0;
}