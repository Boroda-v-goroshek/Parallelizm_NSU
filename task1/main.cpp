#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

template <typename T>
T calculateSinusSum(size_t n) {
    std::vector<T> array(n);
    T period = 2 * M_PI; 
    for (size_t i = 0; i < n; ++i) {
        array[i] = std::sin(i * period / n);
    }
    
    return std::accumulate(array.begin(), array.end(), T(0));
}

int main() {
    const size_t n = 10000000;
#ifdef USE_DOUBLE
    auto sum = calculateSinusSum<double>(n);
    std::cout << "Сумма для типа double: " << sum << std::endl;
#elif defined(USE_FLOAT)
    auto sum = calculateSinusSum<float>(n);
    std::cout << "Сумма для типа float: " << sum << std::endl;
#else
    std::cerr << "Не выбран тип данных." << std::endl;
    return 1;
#endif

    return 0;
}