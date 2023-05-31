add_executable(alu_demo
  "cv_methods_test.cc"
)
target_link_libraries(alu_demo
  PUBLIC
    ${alulibs}
    ${OpenCV_LIBS}
)
