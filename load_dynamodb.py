# ---------------------------------------------------------------------------
# load_dynamodb.py
# Copies every invocation result from the raw CSV files into a DynamoDB table,
# so the results also live in AWS (as planned in the proposal architecture).
#
# - Creates the table "lv-invocations-v2" the first time (on-demand billing,
#   so I only pay per write; 18,000 writes cost a few cents at most).
# - Safe to run again: each call has a unique key, so re-loading the same
#   CSV just overwrites the same items, it never makes duplicates.
#
# Run with:  python3 load_dynamodb.py          (loads all results/raw_*.csv)
#            python3 load_dynamodb.py results/raw_W2_....csv   (one file)
# ---------------------------------------------------------------------------
import csv
import glob
import sys
from decimal import Decimal

import boto3

from config import REGION   # home region: the table lives in eu-north-1

# v2: the key now includes the region, because the same function name exists
# in every region. (v1 "lv-invocations" was single-region and can be deleted.)
TABLE = "lv-invocations-v2"

# Columns stored as numbers (DynamoDB needs Decimal, not float).
NUMBER_COLS = {"hour_of_day", "local_hour", "memory_mb", "invocation_idx", "is_cold", "init_ms",
               "duration_ms", "billed_ms", "max_mem_mb", "client_ms"}

ddb = boto3.client("dynamodb", region_name=REGION)


def ensure_table():
    """Create the table if it does not exist yet, and wait until it is ready."""
    if TABLE in ddb.list_tables()["TableNames"]:
        return
    print(f"Creating table {TABLE} ...")
    ddb.create_table(
        TableName=TABLE,
        # Primary key:
        #   config  (partition key) = region # function name,
        #                             e.g. eu-west-1#lv-py-cpu-128
        #   call_id (sort key)      = window # time # call number
        # So all calls of one configuration sit together, in time order.
        KeySchema=[{"AttributeName": "config", "KeyType": "HASH"},
                   {"AttributeName": "call_id", "KeyType": "RANGE"}],
        AttributeDefinitions=[{"AttributeName": "config", "AttributeType": "S"},
                              {"AttributeName": "call_id", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST")
    ddb.get_waiter("table_exists").wait(TableName=TABLE)


def to_item(row):
    """Turn one CSV row into a DynamoDB item."""
    item = {
        "config": f"{row['region']}#{row['function']}",
        # Zero-padded call number so items sort correctly (007 before 150).
        "call_id": f"{row['window']}#{row['timestamp_utc']}#{int(row['invocation_idx']):03d}",
    }
    for k, v in row.items():
        if v == "" or k == "function":
            continue                    # skip empty values (e.g. init_ms on warm calls)
        item[k] = Decimal(v) if k in NUMBER_COLS else v
    return item


def main():
    files = sys.argv[1:] or sorted(glob.glob("results/raw_*.csv"))
    ensure_table()
    table = boto3.resource("dynamodb", region_name=REGION).Table(TABLE)

    total = 0
    for f in files:
        n = 0
        # batch_writer groups items into batches of 25 and retries any that
        # DynamoDB could not write the first time.
        with table.batch_writer(overwrite_by_pkeys=["config", "call_id"]) as batch:
            with open(f) as fh:
                for row in csv.DictReader(fh):
                    batch.put_item(Item=to_item(row))
                    n += 1
        print(f"{f}: {n} items")
        total += n
    print(f"Done: {total} items written to DynamoDB table {TABLE} ({REGION})")


if __name__ == "__main__":
    main()
