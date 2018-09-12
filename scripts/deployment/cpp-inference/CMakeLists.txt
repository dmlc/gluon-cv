cmake_minimum_required(VERSION 2.8)

project(gluoncv-cpp-inference)

set(CMAKE_CXX_STANDARD 11)

if(NOT MXNET_ROOT)
  set(MXNET_ROOT $ENV{MXNET_ROOT})
endif()

if(NOT MXNET_ROOT)
  message( FATAL_ERROR "MXNET_ROOT not set! You can either set environment variable by export MXNET_ROOT=/path/to/incubator-mxnet or set cmake variable by -DMXNET_ROOT=/path/to/incubator-mxnet" )
endif()

# -- MXNet
include_directories(
  ${MXNET_ROOT}/include
  ${MXNET_ROOT}/cpp-package/include
  ${MXNET_ROOT}/3rdparty/dlpack/include
  ${MXNET_ROOT}/3rdparty/dmlc-core/include
  ${MXNET_ROOT}/3rdparty/mkldnn/include
  ${MXNET_ROOT}/3rdparty/mshadow
  ${MXNET_ROOT}/3rdparty/tvm/nnvm/include
  ${CMAKE_CURRENT_LIST_DIR}/src
)

SET(CMAKE_FIND_LIBRARY_PREFIXES "lib")
find_library(MXNET_LIBS NAMES mxnet HINTS "${MXNET_ROOT}/lib")
message(STATUS MXNET_LIBS: ${MXNET_LIBS})
message(STATUS CURRENT_DIR: ${CMAKE_CURRENT_LIST_DIR})
set(LINKER_LIBS ${MXNET_LIBS})

# -- OpenCV
find_package(OpenCV COMPONENTS core highgui imgproc imgcodecs)
if(NOT OpenCV_FOUND) # if not OpenCV 3.x, then imgcodecs are not found
  message(STATUS "OpenCV imgcodecs missing")
  find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
endif()
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${OpenCV_LIBS})
message(STATUS " OpenCV_LIBS=${OpenCV_LIBS}")
message(STATUS "OpenCV ${OpenCV_VERSION} found (${OpenCV_CONFIG_PATH})")

# -- Targets and Installation
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_LIST_DIR})

add_executable(gluoncv-detect src/detect.cpp)
target_link_libraries(gluoncv-detect ${LINKER_LIBS})

set(EXECS gluoncv-detect)
install(TARGETS ${EXECS} DESTINATION install)
if(APPLE)
  install(FILES ${MXNET_LIBS} DESTINATION install/lib)
else()
 install(FILES ${MXNET_LIBS} DESTINATION install)
endif()
