#include "FATE_ABY_millionaire_prob_test.h"

share *BuildMillionaireProbCircuit(share *s_alice, share *s_bob,
                                   BooleanCircuit *bc) {

    share *out;

    /** Calling the greater than equal function in the Boolean circuit class.*/
    out = bc->PutGTGate(s_alice, s_bob);

    return out;
}


int32_t my_test_millionaire(u_int32_t money, e_role role, const std::string &address, uint16_t port, seclvl seclvl,
                            uint32_t bitlen, uint32_t nthreads, e_mt_gen_alg mt_alg, e_sharing sharing) {


    ABYParty *party = new ABYParty(role, address, port, seclvl, bitlen, nthreads,
                                   mt_alg);
    /*
     * role: the role of the party, SERVER or CLIENT，如果是SERVER，则会在本地端口上建立一个socket，监听来自CLIENT的连接，如果是CLIENT，则会连接到SERVER的socket上
     * */

    std::vector<Sharing *> &sharings = party->GetSharings();

    Circuit *circ = sharings[sharing]->GetCircuitBuildRoutine();

    share *s_alice_money, *s_bob_money, *s_out;

    uint32_t output;

    if (role == SERVER) {    // SERVER == 0, CLIENT == 1
        s_alice_money = circ->PutDummyINGate(bitlen);
        s_bob_money = circ->PutINGate(money, bitlen, SERVER);   // SEVER is the role of Bob
    } else { //role == CLIENT
        s_alice_money = circ->PutINGate(money, bitlen, CLIENT);   // CLIENT is the role of Alice
        s_bob_money = circ->PutDummyINGate(bitlen);
    }

    s_out = BuildMillionaireProbCircuit(s_alice_money, s_bob_money,
                                        (BooleanCircuit *) circ);

    s_out = circ->PutOUTGate(s_out, ALL);

    party->ExecCircuit();

    output = s_out->get_clear_value<uint32_t>();

    std::cout << "Testing Millionaire's Problem in " << get_sharing_name(sharing)
              << " sharing: " << std::endl;
    if (role == SERVER) {
        std::cout << "\nBob's Money:\t" << money;
        std::cout << "\nCircuit Result:\t" << (output ? ALICE : BOB);
        std::cout << "\nOutput\t" << output;
    } else {
        std::cout << "\nAlice's Money:\t" << money;
        std::cout << "\nCircuit Result:\t" << (output ? ALICE : BOB);
        std::cout << "\nOutput\t" << output;

    }

    delete party;
    if (output == 1) {
        return 1;
    } else if (output == 0) {
        return 0;
    } else {
        return -1;
    }
}


extern "C"
int bob(uint32_t money , const char *address, uint16_t port){
    return my_test_millionaire(money, SERVER, address, port, get_sec_lvl(128), 32, 1, MT_OT, S_YAO);
}

extern "C"
int alice(uint32_t money , const char *address, uint16_t port){
    return my_test_millionaire(money, CLIENT, address, port, get_sec_lvl(128), 32, 1, MT_OT, S_YAO);
}

extern "C"
const char* string_test(const char *str){
    std::cout << str << std::endl;
    std::cout << "used!"<< std::endl;
    return "hello world";
}


extern "C" MY_LIB_API
int test() {
    uint32_t bitlen = 32, secparam = 128, nthreads = 1;
    uint16_t port = 7766;
    const char * address = "192.168.210.135";


    uint32_t bob_money, alice_money;
    srand(time(NULL));
    bob_money = rand() % 100;
    alice_money = rand() % 100;
    while (bob_money <= alice_money) {
        bob_money = rand() % 100;
        alice_money = rand() % 100;
    }

    std::cout << "True Result: " << (bob_money > alice_money ? "BOB" : "ALICE") << std::endl;
    pid_t pid = fork();
    if (pid < 0) {
        std::cout << "fork error" << std::endl;
        exit(1);
    }
    if (pid == 0) {
        // 子进程
        bob(bob_money, "0.0.0.0", port);
    } else {
        // 父进程
        alice(alice_money, address, port);
    }
    return 0;
}

int main(){
    test();
}