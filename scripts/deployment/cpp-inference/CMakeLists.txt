cmake_minimum_required(VERSION 2.8.1.2)

project(gluoncv-cpp-inference)

set(CMAKE_CXX_STANDARD 11)

if(NOT MXNET_ROOT)
set(MXNET_ROOT $ENV{MXNET_ROOT})
endif()

if(NOT MXNET_ROOT)
message( FATAL_ERROR "MXNET_ROOT not set! You can either set environment variable by export MXNET_ROOT=/path/to/incubator-mxnet or set cmake variable by -DMXNET_ROOT=/path/to/incubator-mxnet" )
endif()


include_directories(
  ${MXNET_ROOT}/include    
  ${MXNET_ROOT}/cpp-package/include
  ${MXNET_ROOT}/3rdparty/dlpack/include
  ${MXNET_ROOT}/3rdparty/dmlc-core/include
  ${MXNET_ROOT}/3rdparty/mkldnn/include
  ${MXNET_ROOT}/3rdparty/mshadow
  ${MXNET_ROOT}/3rdparty/tvm/nnvm/include
)


find_library(MXNET_LIB NAMES mxnet HINTS "${MXNET_ROOT}/lib")
message(STATUS MXNET_LIB: ${MXNET_LIB})
message(STATUS CURRENT_DIR: ${CMAKE_CURRENT_LIST_DIR})

set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_LIST_DIR})

add_executable(gluoncv_detector detector.cpp)
target_link_libraries(gluoncv_detector ${MXNET_LIB})

set(EXECS gluoncv_detector)
install(TARGETS ${EXECS} DESTINATION install)
install(FILES ${MXNET_LIB} DESTINATION install/lib)