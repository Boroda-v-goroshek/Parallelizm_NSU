#include <iostream>

using namespace std;
int main(){

    #pragma omp parallel
    {
        cout << "Aboba" << endl;
    }

    return 0;
}