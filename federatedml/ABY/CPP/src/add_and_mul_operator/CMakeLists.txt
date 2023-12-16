add_executable(FATE_ABY_add_and_mul_operator add_and_mul_operator.cpp
                                    add_and_mul_operator.h
                                    common/circuit.cpp)

target_link_libraries(FATE_ABY_add_and_mul_operator ABY::aby ENCRYPTO_utils::encrypto_utils)

add_library(FATE_ABY_add_and_mul_operator_lib SHARED add_and_mul_operator.cpp
                                             add_and_mul_operator.h
                                                common/circuit.cpp)

target_link_libraries(FATE_ABY_add_and_mul_operator_lib ABY::aby ENCRYPTO_utils::encrypto_utils)


