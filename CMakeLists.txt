cmake_minimum_required(VERSION 3.5)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/modules/")

set(CMAKE_CXX_STANDARD 11)

# find Eigen
find_package(Eigen REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# find OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# find Magnum
find_package(Magnum REQUIRED
  MeshTools
  Primitives
  Shaders
  SceneGraph
  Sdl2Application)
include_directories(${SDL2_INCLUDE_DIR})

# find ImGui
find_library(IMGUI_LIB ImGui)
find_library(MAGNUM_IMGUI_LIB MagnumImGui)

# libraries to link
set(GRAPHICS_LIBS
  Magnum::MeshTools
  Magnum::Primitives
  Magnum::Application
  Magnum::Magnum
  Magnum::Shaders
  Magnum::SceneGraph
  ${IMGUI_LIB}
  ${MAGNUM_IMGUI_LIB})

add_subdirectory(src)

