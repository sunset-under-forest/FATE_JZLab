from federatedml.ABY.operator.vector_operator64 import vector_add_operator_client, vector_mul_operator_client, vector_operator_execute
address = "127.0.0.1"
port = 7766


vec = [1, 2, 3, 4, 5 , 6 ,7]
vec_len = len(vec)

result_vec , result_type = vector_operator_execute(vector_add_operator_client(), vec, address, port)
print(result_vec[:vec_len], result_type)

result_vec , result_type = vector_operator_execute(vector_mul_operator_client(), vec, address, port)
print(result_vec[:vec_len], result_type)



