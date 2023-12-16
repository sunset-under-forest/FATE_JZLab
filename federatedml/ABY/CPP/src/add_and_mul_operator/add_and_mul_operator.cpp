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
