##add_executable(test common/test.cpp
##        common/test.h)
#add_executable(client_test client_test.cpp
#        client_test.h)
#add_executable(server_test server_test.cpp
#        server_test.h)
#
#
##target_link_libraries(test ABY::aby ENCRYPTO_utils::encrypto_utils)
#target_link_libraries(client_test ABY::aby ENCRYPTO_utils::encrypto_utils)
#target_link_libraries(server_test ABY::aby ENCRYPTO_utils::encrypto_utils)
#
#
##add_library(test_lib SHARED common/test.cpp
##        common/test.h)
##target_link_libraries(test_lib ABY::aby ENCRYPTO_utils::encrypto_utils)
#
#add_library(client_test_lib SHARED client_test.cpp
#        client_test.h)
#target_link_libraries(client_test_lib ABY::aby ENCRYPTO_utils::encrypto_utils)
#
#add_library(server_test_lib SHARED server_test.cpp
#        server_test.h)
#target_link_libraries(server_test_lib ABY::aby ENCRYPTO_utils::encrypto_utils)
#
#add_executable(share_conversion_test share_conversion_test.cpp
#        common/test_const.cpp)
#
#target_link_libraries(share_conversion_test ABY::aby ENCRYPTO_utils::encrypto_utils)
#
#add_executable(relu_conversion_test relu_conversion_test.cpp
#        common/test_const.cpp)
#
#target_link_libraries(relu_conversion_test ABY::aby ENCRYPTO_utils::encrypto_utils)

add_executable(relu_conversion_test relu_conversion_test.cpp
        common/test_const.cpp)

target_link_libraries(relu_conversion_test ABY::aby ENCRYPTO_utils::encrypto_utils)
