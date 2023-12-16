from typing import Any, Callable, List, Tuple
from federatedml.ABY.operator.constant import DLL_PATH
import ctypes
import os


vector_operator_dll = ctypes.CDLL(os.path.join(DLL_PATH, "libFATE_ABY_add_and_mul_operator_lib.so"))

return_type = ctypes.POINTER(ctypes.c_uint32)

vector_operator_dll.aby_add_vector_operator_server.restype = return_type
vector_operator_dll.aby_add_vector_operator_client.restype = return_type
vector_operator_dll.aby_mul_vector_operator_server.restype = return_type
vector_operator_dll.aby_mul_vector_operator_client.restype = return_type

vector_operator_dll.delete_vector.restype = None


# 返回一个函数 uint32_t * aby_add_vector_operator_server(uint32_t * vector, uint32_t vector_length, const char *address, uint16_t port){
#     std::vector<uint32_t> vector_vector(vector, vector + vector_length);
def vector_operator(operator:str, role:str) -> Callable[[ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint16], ctypes.POINTER(ctypes.c_uint32)]:
    """
    :param operator: "add" or "mul"
    :param role: "client" or "server"
    :return: an aby vector operator function in ctypes
    """
    operator = operator.lower().strip()
    role = role.lower().strip()

    if operator == "add":
        if role == "client":
            return vector_operator_dll.aby_add_vector_operator_client
        elif role == "server":
            return vector_operator_dll.aby_add_vector_operator_server
        else:
            raise ValueError("role should be client or server")
    elif operator == "mul":
        if role == "client":
            return vector_operator_dll.aby_mul_vector_operator_client
        elif role == "server":
            return vector_operator_dll.aby_mul_vector_operator_server
        else:
            raise ValueError("role should be client or server")

    else:
        raise ValueError("operator should be add or mul")

def vector_add_operator(role: str) -> Callable[[ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint16], ctypes.POINTER(ctypes.c_uint32)]:
    """
    :param role: "client" or "server"
    :return: an aby vector add function in ctypes
    """
    return vector_operator("add", role)

def vector_mul_operator(role:str) -> Callable[[ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint16], ctypes.POINTER(ctypes.c_uint32)]:
    """
    :param role: "client" or "server"
    :return: an aby vector mul function in ctypes
    """
    return vector_operator("mul", role)

def vector_add_operator_client() -> Callable[[ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint16], ctypes.POINTER(ctypes.c_uint32)]:
    """
    :return: an aby vector add function in ctypes
    """
    return vector_add_operator("client")

def vector_add_operator_server() -> Callable[[ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint16], ctypes.POINTER(ctypes.c_uint32)]:
    """
    :return: an aby vector add function in ctypes
    """
    return vector_add_operator("server")

def vector_mul_operator_client() -> Callable[[ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint16], ctypes.POINTER(ctypes.c_uint32)]:
    """
    :return: an aby vector mul function in ctypes
    """
    return vector_mul_operator("client")

def vector_mul_operator_server() -> Callable[[ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint16], ctypes.POINTER(ctypes.c_uint32)]:
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
    vector_c_type = (ctypes.c_uint32 * vector_length)(*vector)
    address_c_type = ctypes.c_char_p(address.encode("utf-8"))
    port_c_type = ctypes.c_uint16(port)
    result_c_type = operator(vector_c_type, vector_length, address_c_type, port_c_type)

    # TODO:
    result_type = return_type
    return result_c_type, result_type

def vector_delete(vector:ctypes.POINTER(ctypes.c_uint32)) -> None:
    """
    :param vector: a vector of uint32_t
    :return: None
    """
    vector_operator_dll.delete_vector(vector)

