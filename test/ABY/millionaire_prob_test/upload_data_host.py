import os
import argparse
from pipeline.backend.pipeline import PipeLine

DATA_BASE = "/home/lab/federated_learning/fate/from_src_build/FATE/aby_fate_test/data"  # 数据的存放的目录


def main(data_base=DATA_BASE):
    # parties config
    host = 10000

    # partition for data storage
    partition = 4

    data = {"name": "millionaire_prob_test_host", "namespace": f"test"}

    pipeline_upload = PipeLine().set_initiator(role="host", party_id=host).set_roles(host=host)

    pipeline_upload.add_upload_data(file=os.path.join(data_base, "millionaire_prob_test_host.csv"),
                                    table_name=data["name"],             # table name
                                    namespace=data["namespace"],         # namespace
                                    head=1, partition=partition)               # data info

    pipeline_upload.upload(drop=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("--base", "-b", type=str,
                        help="data base, path to directory that contains examples/data")

    args = parser.parse_args()
    if args.base is not None:
        main(args.base)
    else:
        main()
