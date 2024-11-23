cmake_minimum_required(VERSION 3.10)

project(MultiAgentCoverageControl CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  include
  ${YAML_CPP_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# add_library(${PROJECT_NAME}
# )

# target_link_libraries(${PROJECT_NAME}
#   ${YAML_CPP_LIBRARIES}
#   ${OpenCV_LIBRARIES}
# )

add_executable(weight_field_generator
  src/weight_field_generator.cpp
)

target_link_libraries(weight_field_generator
  # ${PROJECT_NAME}
  ${YAML_CPP_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
