cmake_minimum_required (VERSION 2.8.6)
project (visualize)

# Set up include directories
include_directories (include)

# Specify output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(QGLViewer REQUIRED QtCore QtGui)
find_package(Qt4)
set(QT_USE_QTXML 1)
include(${QT_USE_FILE})

add_executable (visualize
	visualize.cpp)

target_link_libraries(visualize ${QT_LIBRARIES})
target_link_libraries(visualize ${QGLViewer_LIBS})