import logging
import yaml
import boto
import pandas as pd
import os


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

def load_data_by_key(key=None, bucket=None, bucket_keys=None, H=False):
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
        df = pd.DataFrame.from_csv(bucket.get_key(bucket_keys[str(key)]))
        df.reset_index(inplace=True)
    else:
        print "ERROR <= Update Logging => NULL, KEY"
        print load_data_by_key.__doc__
    return df

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
    conn = boto.connect_s3(aws_access_key_id=aws['key1'],aws_secret_access_key=aws['key2'])
    bucket = conn.get_bucket(aws['lumos_bucket'])
    return bucket

if __name__ == "__main__":
    '''
    CODE EXECUTES WHEN IMPORTED W/O MAIN
    '''
    aws_bucket_keys = get_keys_for_bucket()
    bucket = connect_bucket()
    load_data_by_key(key='game_id_key',bucket=bucket, bucket_keys=aws_bucket_keys,H=True)
