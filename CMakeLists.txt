cmake_minimum_required(VERSION 3.15)
project(PCA_Library CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(pcalib)

add_custom_target(install_all
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target install
        COMMENT "Running install for pca_lib"
)

# Эта цель будет автоматически выполняться после сборки
add_dependencies(install_all pca_lib)
