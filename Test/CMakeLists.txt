add_executable(Test-Render Test-Render.cpp)
target_include_directories(Test-Render PUBLIC ../FF/include/)
target_link_libraries(Test-Render PUBLIC FF GL GLU GLEW SDL2)