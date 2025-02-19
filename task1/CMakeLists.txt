cmake_minimum_required(VERSION 3.10)
project(FloatDoubleExample)

# Определяем исполняемый файл для float
add_executable(float main.cpp)
target_compile_definitions(float PRIVATE USE_FLOAT)

# Определяем исполняемый файл для double
add_executable(double main.cpp)
target_compile_definitions(double PRIVATE USE_DOUBLE)

# Опционально: добавим цель clean, если нужно
set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "float;double")