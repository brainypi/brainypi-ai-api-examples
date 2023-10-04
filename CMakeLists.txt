cmake_minimum_required(VERSION 3.0)
project(ai_api_client_examples)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lpistache")


find_package(PkgConfig)
pkg_check_modules(Pistache REQUIRED IMPORTED_TARGET libpistache)
pkg_check_modules(RapidJSON REQUIRED IMPORTED_TARGET RapidJSON)


find_package(OpenCV REQUIRED)
if (${OpenCV_VERSION} VERSION_LESS 4.0.0)
  message(FATAL_ERROR "OpenCV version must be >= 4.0.0")
endif()

add_subdirectory(cpp)
