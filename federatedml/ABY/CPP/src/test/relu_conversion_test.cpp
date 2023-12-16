//
// Created by 0ne_bey0nd on 2023/10/24.
//

#include "relu_conversion_test.h"

uint32_t add_rounds, gt_rounds, bitlen, nvals = 7;
uint32_t *zero_check_in_finite_field_val, *zero_val;
std::vector<uint32_t> x1 = {1, 2, 3, 4, 5, 6, 7};
std::vector<uint32_t> x2 = {2, (uint32_t) -4, 6, (uint32_t) -8, 10, (uint32_t) -12, 14};

e_role role;
const char *address = "127.0.0.1";
uint16_t port = 6677;
ABYParty *party;
Circuit *arith_circ, *bool_circ, *yao_circ;
share *s_X1, *s_X2, *s_X, *s_Y;

uint32_t *output, output_bitlen, out_length;


// ʵ��relu�İ�ȫ�෽����
// y = relu(x1 + x2) = max(x1 + x2, 0)
// Ҳ�������������ݶ����x1 + x2��Ȼ���ò�����yao�ݶ��·����max(x1 + x2, 0)

void initialization() {
    // �����������ظ�ʹ�õĲ�����ȡ��������������������ر�����Ӱ��
    party = new ABYParty(role, address, port);
    bitlen = 32;


    // ��Ϊ���ж��Ƿ����0��������Ҫһ��0����������������ķݶ��Ǵ�����һ��2**l���������еģ����Բ���ֱ����0
    // ����ͨ���ж��Ƿ����INT_MAX��ʵ��
    zero_check_in_finite_field_val = new uint32_t[nvals];
    zero_val = new uint32_t[nvals];
    //    memset(zero_check_in_finite_field_val, (int)INT32_MAX, nvals * sizeof(uint32_t));    ��Ҫʹ��memset����Ϊmemset�ǰ��ֽڸ�ֵ�ģ������ǰ�һ��һ��uint32_t��ֵ��
    for (int i = 0; i < nvals; i++) {
        zero_val[i] = 0;
        zero_check_in_finite_field_val[i] = INT32_MAX;
    }

}

share *BuildAddCircuit(share *s_a, share *s_b, Circuit *circ) {
    share *out;

//    std::cout << get_sharing_name(circ->GetContext()) << std::endl;

    out = circ->PutADDGate(s_a, s_b);
    return out;
}

share *BuildReluCircuit(share *s_x, Circuit *circ) {
    share *out;
    share *zero_check_in_finite_field, *zero;
//    uint32_t  bitlen = s_x->get_bitlength();

/*

    // 1��int�ڴ��еĶ����Ʊ�ʾ
    int test_int = 1;
    unsigned  char *test_int_ptr = (unsigned  char *) &test_int;
    std::cout << "test_int_ptr = " << std::endl;
    // ֱ�Ӵ�ӡ�ڴ棬ÿ��bit
    for (int i = 0; i < sizeof(int); ++i) {
        for (int j = 0; j < 8; ++j) {
            std::cout << ((test_int_ptr[i] >> j) & 1);
        }
        std::cout << " ";
    }

    std::cout << std::endl;

    // 1��uint32_t�ڴ��еĶ����Ʊ�ʾ

    uint32_t test_uint32_t = 1;
    unsigned  char *test_uint32_t_ptr = (unsigned  char *) &test_uint32_t;
    std::cout << "test_uint32_t_ptr = " << std::endl;
    // ֱ�Ӵ�ӡ�ڴ棬ÿ��bit
    for (int i = 0; i < sizeof(uint32_t); ++i) {
        for (int j = 0; j < 8; ++j) {
            std::cout << ((test_uint32_t_ptr[i] >> j) & 1);
        }
        std::cout << " ";
    }

    std::cout << std::endl;
*/

    zero = circ->PutSIMDCONSGate(s_x->get_nvals(), zero_val, bitlen);
    zero_check_in_finite_field = circ->PutSIMDCONSGate(s_x->get_nvals(), zero_check_in_finite_field_val, bitlen);

//    std::cout<< get_circuit_type_name(circ->GetCircuitType()) <<std::endl;

    out = circ->PutMUXGate(zero, s_x, circ->PutGTGate(s_x, zero_check_in_finite_field));
    // means out = s_X > zero_check_in_finite_field ? zero : s_X
    // Ҳ����˵�ж�s_x1_x2�Ƿ����INT32_MAX�����������˵��s_x1_x2�Ǹ�������ôout����0���������s_x1_x2

    return out;
}

void check_share_val(share *s) {
    s->get_clear_value_vec(&output, &output_bitlen, &out_length);
    auto y = std::vector<uint32_t>(output, output + out_length);

    std::cout << "y = relu(x1 + x2) = max(x1 + x2, 0) = " << std::endl;
    for (auto i: y) {
        std::cout << (int) i << " ";
    }
    std::cout << std::endl;

    delete[]output;
}


void relu_A2Y_share_conversion_circuit(){
    if (role == SERVER) {
        s_X1 = arith_circ->PutSIMDINGate(x1.size(), x1.data(), bitlen, SERVER);
        s_X2 = arith_circ->PutDummySIMDINGate(x2.size(), bitlen);
    } else {
        s_X1 = arith_circ->PutDummySIMDINGate(x1.size(), bitlen);
        s_X2 = arith_circ->PutSIMDINGate(x2.size(), x2.data(), bitlen, CLIENT);
    }

    s_X= arith_circ->PutSUBGate(s_X1, s_X2);

    s_Y = s_X;

    // arith calculation
    for (int i = 0; i < add_rounds; ++i) {
        s_Y = BuildAddCircuit(s_Y, s_X, arith_circ);
    }

    // share conversion to yao
    s_Y = yao_circ->PutA2YGate(s_Y);

    // logic calculation
    for (int i = 0; i < gt_rounds; ++i) {
        s_Y = BuildReluCircuit(s_Y, yao_circ);
    }

    // share conversion back to arithmetic
    s_Y = arith_circ->PutY2AGate(s_Y, bool_circ);

    s_Y = arith_circ->PutOUTGate(s_Y, ALL);
}

void relu_A2B_share_conversion_circuit(){
    if (role == SERVER) {
        s_X1 = arith_circ->PutSIMDINGate(x1.size(), x1.data(), bitlen, SERVER);
        s_X2 = arith_circ->PutDummySIMDINGate(x2.size(), bitlen);
    } else {
        s_X1 = arith_circ->PutDummySIMDINGate(x1.size(), bitlen);
        s_X2 = arith_circ->PutSIMDINGate(x2.size(), x2.data(), bitlen, CLIENT);
    }

    s_X= arith_circ->PutSUBGate(s_X1, s_X2);

    s_Y = s_X;

    // arith calculation
    for (int i = 0; i < add_rounds; ++i) {
        s_Y = BuildAddCircuit(s_Y, s_X, arith_circ);
    }

    // share conversion to bool
    s_Y = bool_circ->PutA2BGate(s_Y,yao_circ);

    // logic calculation
    for (int i = 0; i < gt_rounds; ++i) {
        s_Y = BuildReluCircuit(s_Y, bool_circ);
    }

    // share conversion back to arithmetic
    s_Y = arith_circ->PutB2AGate(s_Y);

    s_Y = arith_circ->PutOUTGate(s_Y, ALL);
}


void relu_pure_bool_circuit(){
    if (role == SERVER) {
        s_X1 = bool_circ->PutSIMDINGate(x1.size(), x1.data(), bitlen, SERVER);
        s_X2 = bool_circ->PutDummySIMDINGate(x2.size(), bitlen);
    } else {
        s_X1 = bool_circ->PutDummySIMDINGate(x1.size(), bitlen);
        s_X2 = bool_circ->PutSIMDINGate(x2.size(), x2.data(), bitlen, CLIENT);
    }

    s_X = bool_circ->PutSUBGate(s_X1, s_X2);

    s_Y = s_X;

    // arith calculation
    for (int i = 0; i < add_rounds; ++i) {
        s_Y = BuildAddCircuit(s_Y, s_X, bool_circ);
    }

    // logic calculation
    for( int i = 0; i < gt_rounds; ++i){
        s_Y = BuildReluCircuit(s_Y, bool_circ);
    }

    s_Y = bool_circ->PutOUTGate(s_Y, ALL);
}

void relu_pure_yao_circuit(){
    if (role == SERVER) {
        s_X1 = yao_circ->PutSIMDINGate(x1.size(), x1.data(), bitlen, SERVER);
        s_X2 = yao_circ->PutDummySIMDINGate(x2.size(), bitlen);
    } else {
        s_X1 = yao_circ->PutDummySIMDINGate(x1.size(), bitlen);
        s_X2 = yao_circ->PutSIMDINGate(x2.size(), x2.data(), bitlen, CLIENT);
    }

    s_X = yao_circ->PutSUBGate(s_X1, s_X2);

    s_Y = s_X;

    // arith calculation
    for (int i = 0; i < add_rounds; ++i) {
        s_Y = BuildAddCircuit(s_Y, s_X, yao_circ);
    }

//    std::cout <<"hey?" << std::endl;

    // logic calculation
    for( int i = 0; i < gt_rounds; ++i){
        s_Y = BuildReluCircuit(s_Y, yao_circ);
    }

    s_Y = yao_circ->PutOUTGate(s_Y, ALL);

}


// ���뺯��ָ��
void relu_test(void (*relu_test_circuit)()) {
    arith_circ = party->GetSharings()[S_ARITH]->GetCircuitBuildRoutine();
    bool_circ = party->GetSharings()[S_BOOL]->GetCircuitBuildRoutine();
    yao_circ = party->GetSharings()[S_YAO]->GetCircuitBuildRoutine();

    relu_test_circuit();

    party->ExecCircuit();

    // output result
//    check_share_val(s_Y);

    // reset circuit
    party->Reset();

}


void relu_A2Y_share_conversion_test() {
    relu_test(relu_A2Y_share_conversion_circuit);
}

void relu_pure_bool_test() {
    relu_test(relu_pure_bool_circuit);
}

void relu_pure_yao_test() {
    relu_test(relu_pure_yao_circuit);
}

void relu_A2B_share_conversion_test() {
    relu_test(relu_A2B_share_conversion_circuit);
}


int32_t
read_test_options(int32_t *argcp, char ***argvp, e_role *role, uint32_t *add_rounds, uint32_t *gt_rounds) {

    uint32_t int_role = 0;

    parsing_ctx options[] =
            {
                    {(void *) &int_role,  T_NUM, "r", "Role: 0/1",  true, false},
                    {(void *) add_rounds, T_NUM, "a", "add rounds", true, false},
                    {(void *) gt_rounds,  T_NUM, "g", "gt rounds",  true, false}
            };


    if (!parse_options(argcp, argvp, options,
                       sizeof(options) / sizeof(parsing_ctx))) {
        print_usage(*argvp[0], options, sizeof(options) / sizeof(parsing_ctx));
        std::cout << "Exiting" << std::endl;
        exit(0);
    }

    assert(int_role < 2);
    *role = (e_role) int_role;

    // TODO: ����
    assert(*add_rounds < 10001);
    assert(*gt_rounds < 10001);

    //delete options;

    return 1;
}


int main(int argc, char **argv) {


    read_test_options(&argc, &argv, &role, &add_rounds, &gt_rounds);
    std::cout << "role = " << get_role_name(role) << std::endl;
    std::cout << "add_rounds = " << add_rounds << std::endl;
    std::cout << "gt_rounds = " << gt_rounds << std::endl;


    // initialization
    initialization();


    // ��ʱ
    auto start = std::chrono::steady_clock::now();

    relu_A2Y_share_conversion_test();

    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double> relu_A2Y_share_conversion_test_elapsed_seconds = end - start;



    // ��ʱ
    start = std::chrono::steady_clock::now();

    relu_A2B_share_conversion_test();

    end = std::chrono::steady_clock::now();

    std::chrono::duration<double> relu_A2B_share_conversion_test_elapsed_seconds = end - start;

    // ��ʱ
    start = std::chrono::steady_clock::now();

    relu_pure_bool_test();

    end = std::chrono::steady_clock::now();

    std::chrono::duration<double> relu_pure_bool_test_elapsed_seconds = end - start;

    // ��ʱ
    start = std::chrono::steady_clock::now();

    relu_pure_yao_test();

    end = std::chrono::steady_clock::now();

    std::chrono::duration<double> relu_pure_yao_test_elapsed_seconds = end - start;

    std::cout << "relu_A2Y_share_conversion_test_elapsed_seconds = " << relu_A2Y_share_conversion_test_elapsed_seconds.count()
              << "s" << std::endl;
    std::cout << "relu_A2B_share_conversion_test_elapsed_seconds = " << relu_A2B_share_conversion_test_elapsed_seconds.count() << "s"
              << std::endl;
    std::cout << "relu_pure_bool_test_elapsed_seconds = " << relu_pure_bool_test_elapsed_seconds.count() << "s"
              << std::endl;
    std::cout << "relu_pure_yao_test_elapsed_seconds = " << relu_pure_yao_test_elapsed_seconds.count() << "s"
              << std::endl;


    delete[]zero_check_in_finite_field_val;
    delete[]zero_val;
    delete party;
    return 0;
}

