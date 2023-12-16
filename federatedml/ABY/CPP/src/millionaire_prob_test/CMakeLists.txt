add_executable(FATE_ABY_millionaire_prob_test FATE_ABY_millionaire_prob_test.cpp
        FATE_ABY_millionaire_prob_test.h)
target_link_libraries(FATE_ABY_millionaire_prob_test ABY::aby ENCRYPTO_utils::encrypto_utils)
add_library(FATE_ABY_millionaire_prob_test_lib SHARED FATE_ABY_millionaire_prob_test.cpp
        FATE_ABY_millionaire_prob_test.h)
target_link_libraries(FATE_ABY_millionaire_prob_test_lib ABY::aby ENCRYPTO_utils::encrypto_utils)
