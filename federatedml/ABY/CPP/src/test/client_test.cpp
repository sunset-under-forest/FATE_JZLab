//
// Created by 0ne_bey0nd on 2023/10/18.
//

#include "client_test.h"

int main(int argc, char **argv) {
    e_role role = CLIENT;

    ABYParty *party = new ABYParty(role);

    std::vector<Sharing *> sharings = party->GetSharings();

    std::cout << sharings[S_YAO]->sharing_type() << std::endl;

    Circuit *circ = sharings[0]->GetCircuitBuildRoutine();

    u_int32_t bitlen = 32;

    u_int32_t vector_length = 7;
    u_int32_t *client_number_vector = new u_int32_t[vector_length];

    for (int i = 0; i < vector_length; ++i) {
        client_number_vector[i] = rand();
    }


    share *client_number_share, *server_number_share;

    client_number_share = circ->PutSIMDINGate(vector_length, client_number_vector, bitlen, CLIENT);
    server_number_share = circ->PutDummySIMDINGate(vector_length, bitlen);


    share *s_out = ((BooleanCircuit *) circ)->PutADDGate(client_number_share, server_number_share);

    s_out = circ->PutOUTGate(s_out, ALL);

    party->ExecCircuit();

    u_int32_t *output, output_bitlen, out_length;

    s_out->get_clear_value_vec(&output, &output_bitlen, &out_length);

    for (int i = 0; i < out_length; ++i) {
        std::cout << output[i] << std::endl;
    }

    std::cout << "hey?" << std::endl;

    delete party;
    delete []client_number_vector;
}