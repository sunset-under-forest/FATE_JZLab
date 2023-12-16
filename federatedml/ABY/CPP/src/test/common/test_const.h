//
// Created by 0ne_bey0nd on 2023/10/18.
//

#ifndef FATE_ABY_TEST_CONST_H
#define FATE_ABY_TEST_CONST_H
#include "ENCRYPTO_utils/crypto/crypto.h"
#include "ENCRYPTO_utils/parse_options.h"
#include <iostream>
#include "abycore/circuit/booleancircuits.h"
#include "abycore/circuit/arithmeticcircuits.h"
#include "abycore/circuit/circuit.h"
#include "abycore/aby/abyparty.h"
#include "abycore/sharing/sharing.h"
#include "chrono"

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



#endif //FATE_ABY_TEST_CONST_H
