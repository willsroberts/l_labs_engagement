import logging
import yaml
import boto
import pandas as pd
import os
from datetime import datetime

rel_path = os.path.expanduser('~')


def get_keys_for_bucket():
    aws_bucket_keys = yaml.load(open(rel_path + '/.aws/aws_buckets.yaml'))
    return aws_bucket_keys


def display_all_keys(bucket=None, H=False):
    '''
    IN: N/A
    OUT: (Type: boto.resultset.ResultSet)
        All keys in module with authorized connection
    '''
    if H:
        print display_all_keys.__doc__
    return bucket.get_all_keys()


def load_data_by_key(key=None, bucket=None, bucket_keys=None, row_limit=None, H=False):
    '''
    IN: Key for S3 Bucket, S3 Bucket, YAML Keys for Bucket, Help Flag Method
    OUT: (Type: Dataframe)
         Either the df of the specified bucket key,
         OR doc string
    '''
    df = None
    if H:
        print load_data_by_key.__doc__
    if key in bucket_keys.keys():
        df = pd.read_csv(bucket.get_key(bucket_keys[str(key)]))
        df.reset_index(inplace=True)
        standardize_user_id(df)
        if row_limit:
            df = df.head(row_limit)
        dfs = build_chunks(df, 10)
    else:
        logging.error('AWS_CREDS: NULL, KEY')
        print load_data_by_key.__doc__
    return dfs


def write_intermed_data_to_s3(bucket=None, bucket_keys=None, df=None, file_name=None, H=False):
    '''
    IN:
    OUT: RETURN
    '''
    if H:
        print write_intermed_data_to_s3.__doc__
    fil = file_name + str(datetime.now())
    df.to_csv(fil, sep='\t')
    try:
        k = bucket.new_key(file_name)
        k.set_contents_from_filename(fil)
    except:
        logging.error('AWS_CREDS: S3 WRITE: ' + Exception.message)
    logging.info('AWS_CREDS: INTERMEDIATE - WRITTEN TO S3')
    return None


def standardize_user_id(df):
    '''
    IN: Dataframe
    OUT: column renamed to convention
    '''
    if "userId" in df.columns:
        df.rename(columns={'userId': 'user_id'}, inplace=True)
        response = "STANDARDIZED"
    response = "ALREADY STANDARD"
    return "USER ID " + response


def build_chunks(df, mod):

    ix_start = df.shape[0] / mod
    ixs = [0]
    dfs = []
    for ix in xrange(mod - 1):
        ixs.append(ix_start * (ix + 1))
    ixs.append(df.shape[0] - 1)

    for i, ix in enumerate(ixs):
        if ix < df.shape[0] - 1 and ix > 0:
            i_usr = df.iloc[ix].user_id
            j_usr = df.iloc[ix].user_id

            while i_usr == j_usr:
                ix += 1
                j_usr = df.iloc[ix].user_id
                ixs[i] = ix

    bounds = []
    for i, c in enumerate(ixs):
        if i < len(ixs) - 1:
            bounds.append((c, ixs[i + 1]))

    for tup in bounds:
        dfs.append(df.iloc[tup[0]:tup[1]])

    return dfs


def connect_bucket(H=False):
    '''
    IN: N/A
    SUM: Method takes environment variables set in external yaml: ~/aws.yaml and connects to bucketbu
         Sets local bucket var
    OUT: (Type: boto.s3.bucket.Bucket)
         S3 Bucket
    '''
    if H:
        print connect_bucket.__doc__
    aws = yaml.load(open(rel_path + '/.aws/aws.yaml'))
    conn = boto.connect_s3(aws_access_key_id=aws['key1'], aws_secret_access_key=aws['key2'])
    bucket = conn.get_bucket(aws['lumos_bucket'])
    return bucket

if __name__ == "__main__":
    '''
    CODE EXECUTES WHEN IMPORTED W/O MAIN
    '''
    aws_bucket_keys = get_keys_for_bucket()
    bucket = connect_bucket()
    load_data_by_key(key='game_id_key', bucket=bucket, bucket_keys=aws_bucket_keys, H=True)
