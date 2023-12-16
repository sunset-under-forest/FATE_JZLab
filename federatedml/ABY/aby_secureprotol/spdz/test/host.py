import numpy as np
from federatedml.ABY.aby_secureprotol.spdz.tensor.fixedpoint_numpy import FixedPointTensor
from federatedml.ABY.aby_secureprotol.spdz import SPDZ
from fate_arch.session import Session

s = Session()
# on host side
guest_party_id = 10000
host_party_id = 10001
host_proxy_ip = "192.168.0.1"  # Generally, it is your current machine IP
federation_id = "spdz_demo"     # choose a common federation id (this should be same in both site)
session_id = "_".join([federation_id, "host", str(host_party_id)])
s.init_computing(session_id)
s.init_federation(federation_id,
                  runtime_conf={
                      "local": {"role": "host", "party_id": host_party_id},
                      "role": {"guest": [guest_party_id], "host": [host_party_id]},
                  },
                  service_conf={"host": host_proxy_ip, "port": 9370})
s.as_global()
partys = s.parties.all_parties
# [Party(role=guest, party_id=10000), Party(role=host, party_id=10001)]


# on host side(assuming PartyId is partys[1]):
data = np.array([[3, 2, 1], [6, 5, 4]])
with SPDZ() as spdz:
    y = FixedPointTensor.from_source("y", data)
    x = FixedPointTensor.from_source("x", partys[0])

    z = (x + y).get()
    t = (x - y).get()
    print(z)
    print(t)
