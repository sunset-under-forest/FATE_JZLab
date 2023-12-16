//
// Created by 0ne_bey0nd on 2023/10/6.
//

#ifndef ABY_FATE_ABY_MILLIONAIRE_PROB_TEST_H
#define ABY_FATE_ABY_MILLIONAIRE_PROB_TEST_H
#include <ENCRYPTO_utils/crypto/crypto.h>
#include <ENCRYPTO_utils/parse_options.h>
#include <iostream>
#include "../../extern/ABY/src/abycore/circuit/booleancircuits.h"
#include "../../extern/ABY/src/abycore/circuit/arithmeticcircuits.h"
#include "../../extern/ABY/src/abycore/circuit/circuit.h"
#include "../../extern/ABY/src/abycore/aby/abyparty.h"
#include "../../extern/ABY/src/abycore/sharing/sharing.h"

#if defined(_MSC_VER)
#define MY_LIB_API __declspec(dllexport) // Microsoft
#elif defined(__GNUC__)
#define MY_LIB_API __attribute__((visibility("default"))) // GCC    # default代表外部可见，hidden代表外部不可见，就是public和private的意思
#else
#define MY_LIB_API // Most compilers export all the symbols by default. We hope for the best here.
#pragma warning Unknown dynamic link import/export semantics.
#endif
#endif //ABY_FATE_ABY_MILLIONAIRE_PROB_TEST_H

#define ALICE   "ALICE"
#define BOB     "BOB"
