//
// Created by 0ne_bey0nd on 2023/10/23.
//

#include "share_conversion_test.h"

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
                        e_sharing sharing = S_BOOL,bool debug_time = false) {
    std::vector<uint32_t> result;
    //    计时
    if (debug_time) {
        auto start = std::chrono::steady_clock::now();

        result = aby_vector_operator(BuildAddCircuit, vector, role, address, port, sharing);

        auto end = std::chrono::steady_clock::now();

        std::chrono::duration<double> elapsed_seconds = end - start;

        std::cout << "role: " << get_role_name(role) << " share type: " << get_sharing_name(sharing) << " add time: "
                  << elapsed_seconds.count() << "s" << std::endl;
    } else {
        result = aby_vector_operator(BuildAddCircuit, vector, role, address, port, sharing);
    }

    return result;
}

std::vector<uint32_t>
aby_mul_vector_operator(std::vector<uint32_t> &vector, e_role role, const char *address, uint16_t port,
                        e_sharing sharing = S_BOOL,bool debug_time = false) {
    std::vector<uint32_t> result;
    //    计时
    if (debug_time) {
        auto start = std::chrono::steady_clock::now();

        result = aby_vector_operator(BuildMulCircuit, vector, role, address, port, sharing);

        auto end = std::chrono::steady_clock::now();

        std::chrono::duration<double> elapsed_seconds = end - start;

        std::cout << "role: " << get_role_name(role) << " share type: " << get_sharing_name(sharing) << " mul time: "
                  << elapsed_seconds.count() << "s" << std::endl;
    } else {
        result = aby_vector_operator(BuildMulCircuit, vector, role, address, port, sharing);
    }

    return result;
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

    std::vector<uint32_t> result_add_arithmetic = aby_add_vector_operator(test_vector, role, address, port, S_ARITH, true);

    std::vector<uint32_t> result_add_bool = aby_add_vector_operator(test_vector, role, address, port, S_BOOL , true );

    std::vector<uint32_t> result_mul_arithmetic = aby_mul_vector_operator(test_vector, role, address, port, S_ARITH, true );

    std::vector<uint32_t> result_mul_bool = aby_mul_vector_operator(test_vector, role, address, port, S_BOOL, true);

    return 0;
}
