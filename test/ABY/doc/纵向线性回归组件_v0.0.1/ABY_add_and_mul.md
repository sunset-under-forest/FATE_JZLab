# ABY框架加法乘法底层算子及相关测试组件嵌入FATE框架记录

承接《百万富翁测速组件嵌入》

首先准备好相关环境变量

```bash
export FEDERATEDML_LIBRARY_PATH=/home/lab/federated_learning/fate/from_src_build/FATE/python/federatedml # {federatedml包路径}
export ABY_LIBRARY_PATH=/home/lab/federated_learning/fate/from_src_build/FATE/python/federatedml/ABY/CPP/extern/ABY # {ABY项目根目录}
export ABY_CPP_SRC_PATH=$FEDERATEDML_LIBRARY_PATH/ABY/CPP
export ABY_COMPONENT_PATH=$FEDERATEDML_LIBRARY_PATH/ABY
export ABY_FATE_TEST_PATH=$FATE_PROJECT_BASE/aby_fate_test
```

## CPP层开发

### 创建目录并编写CPP文件

```bash
cd $ABY_CPP_SRC_PATH
mkdir -p src/add_and_mul_operator/common
echo "add_subdirectory(src/add_and_mul_operator)" >> CMakeLists.txt
vim src/add_and_mul_operator/CMakeLists.txt
vim src/add_and_mul_operator/add_and_mul_operator.cpp
vim src/add_and_mul_operator/add_and_mul_operator.h
vim src/add_and_mul_operator/common/circuit.cpp
vim src/add_and_mul_operator/common/circuit.h
```

#### src/add_and_mul_operator/CMakeLists.txt

```cmake
add_executable(FATE_ABY_add_and_mul_operator add_and_mul_operator.cpp
                                    add_and_mul_operator.h
                                    common/circuit.cpp)

target_link_libraries(FATE_ABY_add_and_mul_operator ABY::aby ENCRYPTO_utils::encrypto_utils)

add_library(FATE_ABY_add_and_mul_operator_lib SHARED add_and_mul_operator.cpp
                                             add_and_mul_operator.h
                                                common/circuit.cpp)

target_link_libraries(FATE_ABY_add_and_mul_operator_lib ABY::aby ENCRYPTO_utils::encrypto_utils)


```

#### src/add_and_mul_operator/add_and_mul_operator.cpp

```cpp
//
// Created by 0ne_bey0nd on 2023/10/18.
//

#include "add_and_mul_operator.h"

//std::vector<uint32_t> aby_add_vector_operator(std::vector<uint32_t> &vector,e_role role, const char *address, uint16_t port, e_sharing sharing = S_BOOL){
//    uint32_t vector_length = vector.size(),bitlen = 32;
//
//    ABYParty *party = new ABYParty(role, address, port);
//
//    std::vector<Sharing *> sharings = party->GetSharings();
//
//    Circuit *circ = sharings[sharing]->GetCircuitBuildRoutine();
//
//    share *server_number_share, *client_number_share;
//
//    if (role == SERVER) {
//        server_number_share = circ->PutSIMDINGate(vector_length, vector.data(), bitlen, SERVER);
//        client_number_share = circ->PutDummySIMDINGate(vector_length, bitlen);
//    } else {
//        server_number_share = circ->PutDummySIMDINGate(vector_length, bitlen);
//        client_number_share = circ->PutSIMDINGate(vector_length, vector.data(), bitlen, CLIENT);
//    }
//
//    share *s_out = BuildAddCircuit(server_number_share, client_number_share, (BooleanCircuit *) circ);
//
//    s_out = circ->PutOUTGate(s_out, ALL);
//
//    party->ExecCircuit();
//
//    uint32_t *output, output_bitlen, out_length;
//
//    s_out->get_clear_value_vec(&output, &output_bitlen, &out_length);
//
//    std::vector<uint32_t> result(output, output + out_length );
//
//    delete party;
//
//    return result;
//}

// 上面的代码只支持加法，乘法的代码跟上面的代码只有“BuildAddCircuit”和“BuildMulCircuit”的区别，现在为了代码的复用性和可拓展，使用函数指针
std::vector<uint32_t>
aby_vector_operator(share *(*BuildCircuitFunc)(share *, share *, BooleanCircuit *), std::vector<uint32_t> &vector,
                    e_role role, const char *address, uint16_t port, e_sharing sharing = S_BOOL) {

    uint32_t vector_length = vector.size(), bitlen = 32;

    ABYParty *party = new ABYParty(role, address, port);

    std::vector<Sharing *> sharings = party->GetSharings();

    Circuit *circ = sharings[sharing]->GetCircuitBuildRoutine();

    share *server_number_share, *client_number_share;

    if (role == SERVER) {
        server_number_share = circ->PutSIMDINGate(vector_length, vector.data(), bitlen, SERVER);
        client_number_share = circ->PutDummySIMDINGate(vector_length, bitlen);
    } else {
        server_number_share = circ->PutDummySIMDINGate(vector_length, bitlen);
        client_number_share = circ->PutSIMDINGate(vector_length, vector.data(), bitlen, CLIENT);
    }

    share *s_out = BuildCircuitFunc(server_number_share, client_number_share, (BooleanCircuit *) circ);

    s_out = circ->PutOUTGate(s_out, ALL);

    party->ExecCircuit();

    uint32_t *output, output_bitlen, out_length;

    s_out->get_clear_value_vec(&output, &output_bitlen, &out_length);

    std::vector<uint32_t> result(output, output + out_length);

    delete party;

    return result;
}

std::vector<uint32_t>
aby_add_vector_operator(std::vector<uint32_t> &vector, e_role role, const char *address, uint16_t port,
                        e_sharing sharing = S_BOOL) {
    return aby_vector_operator(BuildAddCircuit, vector, role, address, port, sharing);
}

std::vector<uint32_t>
aby_mul_vector_operator(std::vector<uint32_t> &vector, e_role role, const char *address, uint16_t port,
                        e_sharing sharing = S_BOOL) {
    return aby_vector_operator(BuildMulCircuit, vector, role, address, port, sharing);
}

// python interface
extern "C" [[maybe_unused]] MY_LIB_API
uint32_t * aby_add_vector_operator_server(uint32_t * vector, uint32_t vector_length, const char *address, uint16_t port){
    std::vector<uint32_t> vector_vector(vector, vector + vector_length);
    std::vector<uint32_t> result = aby_add_vector_operator(vector_vector, SERVER, address, port);
    uint32_t *result_array = new uint32_t[result.size()];
    std::copy(result.begin(), result.end(), result_array);
    return result_array;
}

extern "C" [[maybe_unused]] MY_LIB_API
uint32_t * aby_add_vector_operator_client(uint32_t * vector, uint32_t vector_length, const char *address, uint16_t port){
    std::vector<uint32_t> vector_vector(vector, vector + vector_length);
    std::vector<uint32_t> result = aby_add_vector_operator(vector_vector, CLIENT, address, port);
    uint32_t *result_array = new uint32_t[result.size()];
    std::copy(result.begin(), result.end(), result_array);
    return result_array;
}

extern "C" [[maybe_unused]] MY_LIB_API
uint32_t * aby_mul_vector_operator_server(uint32_t * vector, uint32_t vector_length, const char *address, uint16_t port){
    std::vector<uint32_t> vector_vector(vector, vector + vector_length);
    std::vector<uint32_t> result = aby_mul_vector_operator(vector_vector, SERVER, address, port);
    uint32_t *result_array = new uint32_t[result.size()];
    std::copy(result.begin(), result.end(), result_array);
    return result_array;
}

extern "C" [[maybe_unused]] MY_LIB_API
uint32_t * aby_mul_vector_operator_client(uint32_t * vector, uint32_t vector_length, const char *address, uint16_t port){
    std::vector<uint32_t> vector_vector(vector, vector + vector_length);
    std::vector<uint32_t> result = aby_mul_vector_operator(vector_vector, CLIENT, address, port);
    uint32_t *result_array = new uint32_t[result.size()];
    std::copy(result.begin(), result.end(), result_array);
    return result_array;
}

extern "C" [[maybe_unused]] MY_LIB_API
uint32_t * vector_test(uint32_t * vector, uint32_t vector_length){
//    std::cout << "vector test" << std::endl;
    std::vector<uint32_t> vector_vector(vector, vector + vector_length);
//    for (unsigned int i : vector_vector) {
//        std::cout << i << std::endl;
//    }

    uint32_t *result_array = new uint32_t[vector_length];
//    std::cout << "vector test" << std::endl;
    std::copy(vector_vector.begin(), vector_vector.end(), result_array);
//    std::cout << "vector test" << std::endl;
    return result_array;
}

extern "C" [[maybe_unused]] MY_LIB_API
void delete_vector(uint32_t * array){
    delete []array;
}



int main(int argc, char **argv) {

    e_role role = get_role(&argc, &argv);
    const char *address = "127.0.0.1";
    uint16_t port = 6677;

    std::vector<uint32_t> test_vector = {1, 2, 3, 4, 5, 6, 7};

    std::vector<uint32_t> add_result = aby_add_vector_operator(test_vector, role, address,port);

    std::cout   << "add result:" << std::endl;
    for (unsigned int i : add_result) {
        std::cout << i << std::endl;
    }

    std::vector<uint32_t> mul_result = aby_mul_vector_operator(test_vector, role, address,port);

    std::cout   << "mul result:" << std::endl;
    for (unsigned int i : mul_result) {
        std::cout << i << std::endl;
    }

    std::cout << "hello world" << std::endl;
    return 0;
}

```

#### src/add_and_mul_operator/add_and_mul_operator.h

```cpp
//
// Created by 0ne_bey0nd on 2023/10/18.
//

#ifndef FATE_ABY_ADD_AND_MUL_OPERATOR_H
#define FATE_ABY_ADD_AND_MUL_OPERATOR_H
#include "common/circuit.h"

#endif //FATE_ABY_ADD_AND_MUL_OPERATOR_H

```

#### src/add_and_mul_operator/common/circuit.cpp

```cpp
//
// Created by 0ne_bey0nd on 2023/10/18.
//

#include "add_and_mul_operator.h"

//std::vector<uint32_t> aby_add_vector_operator(std::vector<uint32_t> &vector,e_role role, const char *address, uint16_t port, e_sharing sharing = S_BOOL){
//    uint32_t vector_length = vector.size(),bitlen = 32;
//
//    ABYParty *party = new ABYParty(role, address, port);
//
//    std::vector<Sharing *> sharings = party->GetSharings();
//
//    Circuit *circ = sharings[sharing]->GetCircuitBuildRoutine();
//
//    share *server_number_share, *client_number_share;
//
//    if (role == SERVER) {
//        server_number_share = circ->PutSIMDINGate(vector_length, vector.data(), bitlen, SERVER);
//        client_number_share = circ->PutDummySIMDINGate(vector_length, bitlen);
//    } else {
//        server_number_share = circ->PutDummySIMDINGate(vector_length, bitlen);
//        client_number_share = circ->PutSIMDINGate(vector_length, vector.data(), bitlen, CLIENT);
//    }
//
//    share *s_out = BuildAddCircuit(server_number_share, client_number_share, (BooleanCircuit *) circ);
//
//    s_out = circ->PutOUTGate(s_out, ALL);
//
//    party->ExecCircuit();
//
//    uint32_t *output, output_bitlen, out_length;
//
//    s_out->get_clear_value_vec(&output, &output_bitlen, &out_length);
//
//    std::vector<uint32_t> result(output, output + out_length );
//
//    delete party;
//
//    return result;
//}

// 上面的代码只支持加法，乘法的代码跟上面的代码只有“BuildAddCircuit”和“BuildMulCircuit”的区别，现在为了代码的复用性和可拓展，使用函数指针
std::vector<uint32_t>
aby_vector_operator(share *(*BuildCircuitFunc)(share *, share *, BooleanCircuit *), std::vector<uint32_t> &vector,
                    e_role role, const char *address, uint16_t port, e_sharing sharing = S_BOOL) {

    uint32_t vector_length = vector.size(), bitlen = 32;

    ABYParty *party = new ABYParty(role, address, port);

    std::vector<Sharing *> sharings = party->GetSharings();

    Circuit *circ = sharings[sharing]->GetCircuitBuildRoutine();

    share *server_number_share, *client_number_share;

    if (role == SERVER) {
        server_number_share = circ->PutSIMDINGate(vector_length, vector.data(), bitlen, SERVER);
        client_number_share = circ->PutDummySIMDINGate(vector_length, bitlen);
    } else {
        server_number_share = circ->PutDummySIMDINGate(vector_length, bitlen);
        client_number_share = circ->PutSIMDINGate(vector_length, vector.data(), bitlen, CLIENT);
    }

    share *s_out = BuildCircuitFunc(server_number_share, client_number_share, (BooleanCircuit *) circ);

    s_out = circ->PutOUTGate(s_out, ALL);

    party->ExecCircuit();

    uint32_t *output, output_bitlen, out_length;

    s_out->get_clear_value_vec(&output, &output_bitlen, &out_length);

    std::vector<uint32_t> result(output, output + out_length);

    delete party;

    return result;
}




std::vector<uint32_t>
aby_add_vector_operator(std::vector<uint32_t> &vector, e_role role, const char *address, uint16_t port,
                        e_sharing sharing = S_BOOL) {
    return aby_vector_operator(BuildAddCircuit, vector, role, address, port, sharing);
}

std::vector<uint32_t>
aby_mul_vector_operator(std::vector<uint32_t> &vector, e_role role, const char *address, uint16_t port,
                        e_sharing sharing = S_BOOL) {
    return aby_vector_operator(BuildMulCircuit, vector, role, address, port, sharing);
}

// python interface
extern "C" [[maybe_unused]] MY_LIB_API
uint32_t * aby_add_vector_operator_server(uint32_t * vector, uint32_t vector_length, const char *address, uint16_t port){
    std::vector<uint32_t> vector_vector(vector, vector + vector_length);
    std::vector<uint32_t> result = aby_add_vector_operator(vector_vector, SERVER, address, port);
    uint32_t *result_array = new uint32_t[result.size()];
    std::copy(result.begin(), result.end(), result_array);
    return result_array;
}

extern "C" [[maybe_unused]] MY_LIB_API
uint32_t * aby_add_vector_operator_client(uint32_t * vector, uint32_t vector_length, const char *address, uint16_t port){
    std::vector<uint32_t> vector_vector(vector, vector + vector_length);
    std::vector<uint32_t> result = aby_add_vector_operator(vector_vector, CLIENT, address, port);
    uint32_t *result_array = new uint32_t[result.size()];
    std::copy(result.begin(), result.end(), result_array);
    return result_array;
}

extern "C" [[maybe_unused]] MY_LIB_API
uint32_t * aby_mul_vector_operator_server(uint32_t * vector, uint32_t vector_length, const char *address, uint16_t port){
    std::vector<uint32_t> vector_vector(vector, vector + vector_length);
    std::vector<uint32_t> result = aby_mul_vector_operator(vector_vector, SERVER, address, port);
    uint32_t *result_array = new uint32_t[result.size()];
    std::copy(result.begin(), result.end(), result_array);
    return result_array;
}

extern "C" [[maybe_unused]] MY_LIB_API
uint32_t * aby_mul_vector_operator_client(uint32_t * vector, uint32_t vector_length, const char *address, uint16_t port){
    std::vector<uint32_t> vector_vector(vector, vector + vector_length);
    std::vector<uint32_t> result = aby_mul_vector_operator(vector_vector, CLIENT, address, port);
    uint32_t *result_array = new uint32_t[result.size()];
    std::copy(result.begin(), result.end(), result_array);
    return result_array;
}

extern "C" [[maybe_unused]] MY_LIB_API
uint32_t * vector_test(uint32_t * vector, uint32_t vector_length){
//    std::cout << "vector test" << std::endl;
    std::vector<uint32_t> vector_vector(vector, vector + vector_length);
//    for (unsigned int i : vector_vector) {
//        std::cout << i << std::endl;
//    }

    uint32_t *result_array = new uint32_t[vector_length];
//    std::cout << "vector test" << std::endl;
    std::copy(vector_vector.begin(), vector_vector.end(), result_array);
//    std::cout << "vector test" << std::endl;
    return result_array;
}

extern "C" [[maybe_unused]] MY_LIB_API
void delete_vector(uint32_t * array){
    delete []array;
}

// todo: 64 bits
std::vector<uint64_t>
aby_vector_operator64(share *(*BuildCircuitFunc)(share *, share *, BooleanCircuit *), std::vector<uint64_t> &vector,
                      e_role role, const char *address, uint16_t port, e_sharing sharing = S_BOOL) {

    uint32_t vector_length = vector.size(), bitlen = 64;

    ABYParty *party = new ABYParty(role, address, port,LT, bitlen);

    std::vector<Sharing *> sharings = party->GetSharings();

    Circuit *circ = sharings[sharing]->GetCircuitBuildRoutine();

    share *server_number_share, *client_number_share;

    if (role == SERVER) {
        server_number_share = circ->PutSIMDINGate(vector_length, vector.data(), bitlen, SERVER);
        client_number_share = circ->PutDummySIMDINGate(vector_length, bitlen);
    } else {
        server_number_share = circ->PutDummySIMDINGate(vector_length, bitlen);
        client_number_share = circ->PutSIMDINGate(vector_length, vector.data(), bitlen, CLIENT);
    }

    share *s_out = BuildCircuitFunc(server_number_share, client_number_share, (BooleanCircuit *) circ);

    s_out = circ->PutOUTGate(s_out, ALL);

    party->ExecCircuit();

    uint64_t *output;
    uint32_t  output_bitlen, out_length;

    s_out->get_clear_value_vec(&output, &output_bitlen, &out_length);

    std::vector<uint64_t> result(output, output + out_length);

    delete party;

    return result;
}


std::vector<uint64_t>
aby_add_vector_operator64(std::vector<uint64_t> &vector, e_role role, const char *address, uint16_t port,
                        e_sharing sharing = S_BOOL) {
    return aby_vector_operator64(BuildAddCircuit, vector, role, address, port, sharing);
}

std::vector<uint64_t>
aby_mul_vector_operator64(std::vector<uint64_t> &vector, e_role role, const char *address, uint16_t port,
                        e_sharing sharing = S_BOOL) {
    return aby_vector_operator64(BuildMulCircuit, vector, role, address, port, sharing);
}

extern "C" [[maybe_unused]] MY_LIB_API
uint64_t * aby_add_vector_operator_server64(uint64_t * vector, uint32_t vector_length, const char *address, uint16_t port){
    std::vector<uint64_t> vector_vector(vector, vector + vector_length);
    std::vector<uint64_t> result = aby_add_vector_operator64(vector_vector, SERVER, address, port);
    uint64_t *result_array = new uint64_t[result.size()];
    std::copy(result.begin(), result.end(), result_array);
    return result_array;
}

extern "C" [[maybe_unused]] MY_LIB_API
uint64_t * aby_add_vector_operator_client64(uint64_t * vector, uint32_t vector_length, const char *address, uint16_t port){
    std::vector<uint64_t> vector_vector(vector, vector + vector_length);
    std::vector<uint64_t> result = aby_add_vector_operator64(vector_vector, CLIENT, address, port);
    uint64_t *result_array = new uint64_t[result.size()];
    std::copy(result.begin(), result.end(), result_array);
    return result_array;
}

extern "C" [[maybe_unused]] MY_LIB_API
uint64_t * aby_mul_vector_operator_server64(uint64_t * vector, uint32_t vector_length, const char *address, uint16_t port){
    std::vector<uint64_t> vector_vector(vector, vector + vector_length);
    std::vector<uint64_t> result = aby_mul_vector_operator64(vector_vector, SERVER, address, port);
    uint64_t *result_array = new uint64_t[result.size()];
    std::copy(result.begin(), result.end(), result_array);
    return result_array;
}

extern "C" [[maybe_unused]] MY_LIB_API
uint64_t * aby_mul_vector_operator_client64(uint64_t * vector, uint32_t vector_length, const char *address, uint16_t port){
    std::vector<uint64_t> vector_vector(vector, vector + vector_length);
    std::vector<uint64_t> result = aby_mul_vector_operator64(vector_vector, CLIENT, address, port);
    uint64_t *result_array = new uint64_t[result.size()];
    std::copy(result.begin(), result.end(), result_array);
    return result_array;
}

extern "C" [[maybe_unused]] MY_LIB_API
uint64_t * vector_test64(uint64_t * vector, uint32_t vector_length){
//    std::cout << "vector test" << std::endl;
    std::vector<uint64_t> vector_vector(vector, vector + vector_length);
//    for (unsigned int i : vector_vector) {
//        std::cout << i << std::endl;
//    }

    uint64_t *result_array = new uint64_t[vector_length];
//    std::cout << "vector test" << std::endl;
    std::copy(vector_vector.begin(), vector_vector.end(), result_array);
//    std::cout << "vector test" << std::endl;
    return result_array;

}

extern "C" [[maybe_unused]] MY_LIB_API
void delete_vector64(uint64_t * array){
    delete []array;
}


int main(int argc, char **argv) {

    e_role role = get_role(&argc, &argv);
    const char *address = "127.0.0.1";
    uint16_t port = 6677;

//    std::vector<uint32_t> test_vector = {1, 2, 3, 4, 5, 6, 7};
//
//    std::vector<uint32_t> add_result = aby_add_vector_operator(test_vector, role, address,port);
//
//    std::cout   << "add result:" << std::endl;
//    for (unsigned int i : add_result) {
//        std::cout << i << std::endl;
//    }
//
//    std::vector<uint32_t> mul_result = aby_mul_vector_operator(test_vector, role, address,port);
//
//    std::cout   << "mul result:" << std::endl;
//    for (unsigned int i : mul_result) {
//        std::cout << i << std::endl;
//    }
//
//    std::cout << "hello world" << std::endl;

    std::cout << "hello world64" << std::endl;

    std::vector<uint64_t> test_vector64 = {1, 2, 3, 4, 5, 6, 7};

    std::vector<uint64_t> add_result64 = aby_add_vector_operator64(test_vector64, role, address,port);

    std::cout   << "add result:" << std::endl;
    for (unsigned int i : add_result64) {
        std::cout << i << std::endl;
    }

    std::vector<uint64_t> mul_result64 = aby_mul_vector_operator64(test_vector64, role, address,port);

    std::cout   << "mul result:" << std::endl;
    for (unsigned int i : mul_result64) {
        std::cout << i << std::endl;
    }



    return 0;
}

```

#### src/add_and_mul_operator/common/circuit.h

```cpp
//
// Created by 0ne_bey0nd on 2023/10/18.
//

#ifndef FATE_ABY_CIRCUIT_H
#define FATE_ABY_CIRCUIT_H
#include "ENCRYPTO_utils/crypto/crypto.h"
#include "ENCRYPTO_utils/parse_options.h"
#include <iostream>
#include "abycore/circuit/booleancircuits.h"
#include "abycore/circuit/arithmeticcircuits.h"
#include "abycore/circuit/circuit.h"
#include "abycore/aby/abyparty.h"
#include "abycore/sharing/sharing.h"


#if defined(_MSC_VER)
#define MY_LIB_API __declspec(dllexport) // Microsoft
#elif defined(__GNUC__)
#define MY_LIB_API __attribute__((visibility("default"))) // GCC    # default代表外部可见，hidden代表外部不可见，就是public和private的意思
#else
#define MY_LIB_API // Most compilers export all the symbols by default. We hope for the best here.
#pragma warning Unknown dynamic link import/export semantics.
#endif
int32_t read_test_options(int32_t *argcp, char ***argvp, e_role *role,
                          uint32_t *bitlen, uint32_t *nvals, uint32_t *secparam, std::string *address,
                          uint16_t *port, int32_t *test_op);


e_role get_role(int32_t *argcp, char ***argvp);


share *BuildAddCircuit( share *s_x1, share *s_x2,BooleanCircuit *circ);

share *BuildMulCircuit(share *s_x1, share *s_x2,BooleanCircuit *circ);

#endif //FATE_ABY_CIRCUIT_H

```

### 编译

```bash
cd build
cmake ..
make
```

### 测试

```bash
./src/add_and_mul_operator/FATE_ABY_add_and_mul_operator -r 0 & ./src/add_and_mul_operator/FATE_ABY_add_and_mul_operator -r 1

```

```
(fate_venv) lab@lab-virtual-machine:~/federated_learning/fate/from_src_build/FATE/python/federatedml/ABY/CPP/build$ ./src/add_and_mul_operator/FATE_ABY_add_and_mul_operator -r 0 & ./src/add_and_mul_operator/FATE_ABY_add_and_mul_operator -r 1
[1] 681076
hello world64
hello world64
add result:
add result:
2
2
4
4
6
6
8
8
10
10
12
12
14
14
mul result:
1
4
9
16
25
36
49
mul result:
1
4
9
16
25
36
49
[1]+  Done                    ./src/add_and_mul_operator/FATE_ABY_add_and_mul_operator -r 0
```

测试成功后进入python层开发

## PYTHON层开发

### 复制动态链接库

```bash
cd $ABY_COMPONENT_PATH
cp $ABY_CPP_SRC_PATH/build/src/add_and_mul_operator/libFATE_ABY_add_and_mul_operator_lib.so lib/
```

### 创建python包

```bash
# aby 算子库——对aby底层算子操作提供接口封装
mkdir -p operator
touch operator/__init__.py
# aby 向量算子测试库——这里编写相关的测试组件
mkdir -p vector_operator_test
touch vector_operator_test/__init__.py

```

### 编写算子库python文件

```bash
vim operator/constant.py
vim operator/vector_operator64.py
```

#### operator/constant.py

```python
import os

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
ABY_PKG_PATH = os.path.join(CURRENT_PATH, "..")
DLL_PATH = os.path.join(ABY_PKG_PATH, "lib")

```

#### operator/vector_operator64.py

```python
from typing import Any, Callable, List, Tuple
from federatedml.ABY.operator.constant import DLL_PATH
import ctypes
import os


vector_operator_dll = ctypes.CDLL(os.path.join(DLL_PATH, "libFATE_ABY_add_and_mul_operator_lib.so"))

return_type = ctypes.POINTER(ctypes.c_uint64)

vector_operator_dll.aby_add_vector_operator_server64.restype = return_type
vector_operator_dll.aby_add_vector_operator_client64.restype = return_type
vector_operator_dll.aby_mul_vector_operator_server64.restype = return_type
vector_operator_dll.aby_mul_vector_operator_client64.restype = return_type

vector_operator_dll.delete_vector64.restype = None

def vector_operator(operator:str, role:str) -> Callable[[ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint16], ctypes.POINTER(ctypes.c_uint64)]:
    """
    :param operator: "add" or "mul"
    :param role: "client" or "server"
    :return: an aby vector operator function in ctypes
    """
    operator = operator.lower().strip()
    role = role.lower().strip()

    if operator == "add":
        if role == "client":
            return vector_operator_dll.aby_add_vector_operator_client64
        elif role == "server":
            return vector_operator_dll.aby_add_vector_operator_server64
        else:
            raise ValueError("role should be client or server")
    elif operator == "mul":
        if role == "client":
            return vector_operator_dll.aby_mul_vector_operator_client64
        elif role == "server":
            return vector_operator_dll.aby_mul_vector_operator_server64
        else:
            raise ValueError("role should be client or server")

    else:
        raise ValueError("operator should be add or mul")

def vector_add_operator(role: str) -> Callable[[ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint16], ctypes.POINTER(ctypes.c_uint64)]:
    """
    :param role: "client" or "server"
    :return: an aby vector add function in ctypes
    """
    return vector_operator("add", role)

def vector_mul_operator(role:str) -> Callable[[ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint16], ctypes.POINTER(ctypes.c_uint64)]:
    """
    :param role: "client" or "server"
    :return: an aby vector mul function in ctypes
    """
    return vector_operator("mul", role)

def vector_add_operator_client() -> Callable[[ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint16], ctypes.POINTER(ctypes.c_uint64)]:
    """
    :return: an aby vector add function in ctypes
    """
    return vector_add_operator("client")

def vector_add_operator_server() -> Callable[[ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint16], ctypes.POINTER(ctypes.c_uint64)]:
    """
    :return: an aby vector add function in ctypes
    """
    return vector_add_operator("server")

def vector_mul_operator_client() -> Callable[[ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint16], ctypes.POINTER(ctypes.c_uint64)]:
    """
    :return: an aby vector mul function in ctypes
    """
    return vector_mul_operator("client")

def vector_mul_operator_server() -> Callable[[ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint16], ctypes.POINTER(ctypes.c_uint64)]:
    """
    :return: an aby vector mul function in ctypes
    """
    return vector_mul_operator("server")

def vector_operator_execute(operator:Callable, vector: List[int], address:str, port:int) -> Tuple[
    Any, Any]:
    """
    :param operator: an aby vector operator function in ctypes
    :param vector: a vector of uint32_t
    :param address: an ip address
    :param port: a port
    :return: a vector of uint32_t
    """
    vector_length = len(vector)
    vector_c_type = (ctypes.c_uint64 * vector_length)(*vector)
    address_c_type = ctypes.c_char_p(address.encode("utf-8"))
    port_c_type = ctypes.c_uint16(port)
    result_c_type = operator(vector_c_type, vector_length, address_c_type, port_c_type)

    # TODO:
    result_type = return_type
    return result_c_type, result_type

def vector_delete(vector:ctypes.POINTER(ctypes.c_uint64)) -> None:
    """
    :param vector: a vector of uint32_t
    :return: None
    """
    vector_operator_dll.delete_vector64(vector)


```

### 编写python测试文件查看动态链接库是否可用

```bash
vim vector_operator_test/test_server64.py
vim vector_operator_test/test_client64.py
```

#### vector_operator_test/test_server64.py

```python
from federatedml.ABY.operator.vector_operator64 import vector_add_operator_server, vector_mul_operator_server, vector_operator_execute
address = "127.0.0.1"
port = 7766


vec = [1, 2, 3, 4, 5 , 6 ,7]
vec_len = len(vec)

result_vec , result_type = vector_operator_execute(vector_add_operator_server(), vec, address, port)
print(result_vec[:vec_len], result_type)

result_vec , result_type = vector_operator_execute(vector_mul_operator_server(), vec, address, port)
print(result_vec[:vec_len], result_type)




```

#### vector_operator_test/test_client64.py

```python
from federatedml.ABY.operator.vector_operator64 import vector_add_operator_client, vector_mul_operator_client, vector_operator_execute
address = "127.0.0.1"
port = 7766


vec = [1, 2, 3, 4, 5 , 6 ,7]
vec_len = len(vec)

result_vec , result_type = vector_operator_execute(vector_add_operator_client(), vec, address, port)
print(result_vec[:vec_len], result_type)

result_vec , result_type = vector_operator_execute(vector_mul_operator_client(), vec, address, port)
print(result_vec[:vec_len], result_type)




```

### 执行

```bash
python vector_operator_test/test_server64.py  & python vector_operator_test/test_client64.py
```

```
(fate_venv) lab@lab-virtual-machine:~/federated_learning/fate/from_src_build/FATE/python/federatedml/ABY$ python vector_operator_test/test_server64.py  & python vector_operator_test/test_client64.py
[1] 681230
[2, 4, 6, 8, 10, 12, 14] <class 'federatedml.ABY.operator.vector_operator64.LP_c_ulong'>
[2, 4, 6, 8, 10, 12, 14] <class 'federatedml.ABY.operator.vector_operator64.LP_c_ulong'>
[1, 4, 9, 16, 25, 36, 49] <class 'federatedml.ABY.operator.vector_operator64.LP_c_ulong'>
[1, 4, 9, 16, 25, 36, 49] <class 'federatedml.ABY.operator.vector_operator64.LP_c_ulong'>
[1]+  Done                    python vector_operator_test/test_server64.py

```

没有问题

至此基于ABY框架的向量加法乘法算子嵌入FATE框架已完成。
