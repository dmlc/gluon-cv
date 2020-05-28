cmake_minimum_required(VERSION 3.6)
project(gluoncv_lite)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -ldl -pthread")
#SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
#SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR})
SET(HOME_TVM ${CMAKE_CURRENT_SOURCE_DIR}/tvm)

INCLUDE_DIRECTORIES(${HOME_TVM}/include)
INCLUDE_DIRECTORIES(${HOME_TVM}/3rdparty/dmlc-core/include)
INCLUDE_DIRECTORIES(${HOME_TVM}/3rdparty/dlpack/include)

set(PNG_LIBS libpng.a)
set(Z_LIBS libz.a)
set(JPEG_LIBS libjpeg.a)
link_directories(${PROJECT_SOURCE_DIR}/build)

add_executable(gcv_lite_classify  tvm_runtime_pack.cc src/classification/classification.cpp)
target_link_libraries(gcv_lite_classify    ${CMAKE_DL_LIBS}  ${JPEG_LIBS} ${PNG_LIBS} ${Z_LIBS})

add_executable(gcv_lite_detection  tvm_runtime_pack.cc src/detection/detection.cpp)
target_link_libraries(gcv_lite_detection    ${CMAKE_DL_LIBS}  ${JPEG_LIBS} ${PNG_LIBS} ${Z_LIBS})
