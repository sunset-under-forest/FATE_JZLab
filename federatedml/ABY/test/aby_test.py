

# 读取命令行参数 -r
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--role', required=True, help="role of this party", type=int)
args = parser.parse_args()
role = args.role
SERVER = 0
CLIENT = 1
if role != SERVER and role != CLIENT:
    print("role must be 0 or 1")
    exit(1)

from aby_shared import *
if role == SERVER:
    r = SERVER
else:
    r = CLIENT

party = ABYParty(r)
sharings = party.GetSharings()
circ= sharings[0].GetCircuitBuildRoutine()
alice_money = 7
bob_money = 8
bitlen = 32

if circ.GetRole() == SERVER:
    s_alice_money = circ.PutDummyINGate(bitlen)
    s_bob_money = circ.PutINGate(bob_money,bitlen,SERVER)

else:
    s_alice_money = circ.PutINGate(alice_money,bitlen,CLIENT)
    s_bob_money = circ.PutDummyINGate(bitlen)


s_out = circ.PutGTGate(s_alice_money,s_bob_money)
s_out = circ.PutOUTGate(s_out,ALL)

party.ExecCircuit()
output = s_out.get_clear_value_uint32()
print(output)