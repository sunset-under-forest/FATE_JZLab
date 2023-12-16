import os
import argparse
from pipeline.backend.pipeline import PipeLine


DATA_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")  # 数据存放的目录


def main(data_base=DATA_BASE):
    # parties config
    guest = 10000

    # partition for data storage
    partition = 4

    data = {"name": "hetero_sshe_linr_test_guest", "namespace": f"test"}

    pipeline_upload = PipeLine().set_initiator(role="guest", party_id=guest).set_roles(guest=guest)

    pipeline_upload.add_upload_data(file=os.path.join(data_base, "hetero_sshe_linr_test_guest.csv"),
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
