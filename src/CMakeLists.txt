find_package(Magnum REQUIRED
  MeshTools
  Primitives
  Shaders
  SceneGraph
  Sdl2Application)

add_executable(main main.cpp DrawingPrimitives.cpp)

target_link_libraries(main
  Magnum::MeshTools
  Magnum::Primitives
  Magnum::Application
  Magnum::Magnum
  Magnum::Shaders
  Magnum::SceneGraph
  MagnumImGui)
