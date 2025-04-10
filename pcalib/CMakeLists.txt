cmake_minimum_required(VERSION 3.15)
project(PCA_Library_CPP CXX)

add_library(pca_lib SHARED
        cpp/gauss_solver.cpp
        cpp/centering.cpp
        cpp/matrix_ops.cpp
        cpp/covariance.cpp
        cpp/eigenvalues.cpp
)

if(MSVC)
    target_compile_options(pca_lib PRIVATE /O2)
else()
    target_compile_options(pca_lib PRIVATE -O3 -pthread)
endif()

target_include_directories(pca_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp)

set(INSTALL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/py)

#install(TARGETS pca_lib
#        RUNTIME DESTINATION ${INSTALL_PATH}  # для Windows (.dll)
#        LIBRARY DESTINATION ${INSTALL_PATH}  # для Linux/macOS (.so/.dylib)
#        ARCHIVE DESTINATION ${INSTALL_PATH}  # для статических библиотек (если понадобится)
#)

add_custom_command(TARGET pca_lib POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${INSTALL_PATH}  # Создание каталога установки
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:pca_lib> ${INSTALL_PATH}  # Копирование библиотеки в каталог
        COMMENT "Installing pca_lib to ${INSTALL_PATH}"
)

install(TARGETS pca_lib
        RUNTIME DESTINATION ${INSTALL_PATH}  # Для Windows (.dll)
        LIBRARY DESTINATION ${INSTALL_PATH}  # Для Linux/macOS (.so/.dylib)
        ARCHIVE DESTINATION ${INSTALL_PATH}  # Для статических библиотек (если понадобится)
)
