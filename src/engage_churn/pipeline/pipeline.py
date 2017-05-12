import pandas as pd
from datetime import timedelta,date
# from engage_churn.utilities import aws_bucket_client
import os
# import sys
import pprint
from utilities.aws_bucket_client import display_all_keys, get_keys_for_bucket, load_data_by_key, connect_bucket

# rel_path = os.getcwd()
# sys.path.append(rel_path)
# pprint.pprint(sys.path)

class ECPipeline(object):

    def __init__(self, bucket=None, aws_keys=None):
        self.bucket     = bucket
        self.aws_keys   = None
        self.demo_data  = None
        self.gv_data    = None
        self.gid_data   = None
        self.subs_data  = None
        self.churn_threshold = 30
        self.daily_categories =  {'late_night':[0,1,2,3],
              'early_morning':[4,5,6,7],
              'morning':[8,9,10,11],
              'afternoon':[12,13,14,15],
              'evening':[16,17,18,19],
              'nighttime':[20,21,22,23]}

    def get_s3_bucket(self):
        return self.bucket

    def set_s3_bucket(self, bucket):
        self.bucket = bucket

    def get_aws_keys(self):
        return self.aws_keys

    def set_aws_keys(self,keys):
        self.aws_keys = keys

    def get_demo_data(self):
        '''
        WARN: BE CAUTIOUS RETRIEVING LARGE DS
        '''
        return self.demo_data

    def set_demo_data(self, demo_df):
        self.demo_data = demo_df

    def get_gv_data(self):
        '''
        WARN: BE CAUTIOUS RETRIEVING LARGE DS
        '''
        return self.gv_data

    def set_gv_data(self,gv_df):
        self.gv_data = gv_df

    def get_gid_data(self):
        '''
        WARN: BE CAUTIOUS RETRIEVING LARGE DS
        '''
        return self.gid_data

    def set_gid_data(self,gid_df):
        self.gid_data = gid_df

    def get_churn_threshold(self):
        return self.churn_threshold

    def set_churn_threshold(self,churn_thresh):
        self.churn_threshold = churn_thresh

    def get_subs_data(self):
        return self.subs_data

    def set_subs_data(self, subs_df):
        self.subs_data = subs_df

#####################################################

    def preprocess_demo_df(self, row_limit):
        df = load_data_by_key(key='demo_key',
                              bucket=self.get_s3_bucket(),
                              bucket_keys=self.get_aws_keys())
        df.reset_index(inplace=True)

        if row_limit:
            df = df.head(row_limit)

        #dummify gender
        df_sex = pd.get_dummies(df.gender)
        df = pd.concat([df,df_sex],axis=1)
        df = df.rename(columns={'m':'is_male'})

        #No known individuals above age 117, so filtering those records
        #Unreasonable for age <5 for players
        df = df[df['age'] < 117]
        df = df[df['age'] > 4]

        df.drop(['f','gender'],axis=1,inplace=True)

        print df.describe()[['age','is_male']]
        self.set_demo_data(demo_df=df)
        return "LOGGING (FIX): DEMO DATA SET, SUCCESSFULLY"

    def preprocess_game_id_data(self, row_limit=None):
        df = load_data_by_key(key='game_id_key',
                                bucket=self.get_s3_bucket(),
                                bucket_keys=self.get_aws_keys())
        df.reset_index(inplace=True)

        if row_limit:
            df = df.head(row_limit)

        df['date'] = pd.to_datetime(df.date)
        #Create Column for unique number of games played
        #across all sessions
        df['uniq_games_session'] = df.apply(lambda row : len(set(row['gameId'].split(','))), axis=1)

        #Create column for all hours played by user
        #across all sessions
        q = df.groupby(['userId'])['hour'].apply(lambda x: ','.join(x))
        df = df.join(q, on='userId', how='inner', lsuffix='_gid', rsuffix='_ugdf')

        #Create column for all games played by userId
        #across all sessions
        r = df.groupby(['userId'])['gameId'].apply(lambda x: ','.join(x))
        df = df.join(r, on='userId', how='inner', lsuffix='_gid', rsuffix='_ugpdf')

        df.lpi.fillna(0, inplace=True)

        df.rename(columns={'hour_gid':'session_hours',
                     'gameId_gid':'session_gameIds',
                     'hour_ugdf':'all_user_gaming_hours',
                     'gameId_ugpdf':'all_user_games_plyd',
                     'userId':'user_id'})

        df.drop(['hour_gid', 'date', 'gameId_gid', 'hour_ugdf', 'gameId_ugpdf'],axis=1,inplace=True)

        #Setting dateframe on pipeline object
        self.set_gid_data(gid_df=df)
        return "LOGGING(FIX): GAME ID DATA SET SUCCESSFULLY"

    def preprocess_game_variety_data(self, row_limit=None):
        #if self.get_game_variety_data not none, prompt to make sure
        df = load_data_by_key(key='game_var_key',
                                bucket=self.get_s3_bucket(),
                                bucket_keys=self.get_aws_keys())
        df.reset_index(inplace=True)

        if row_limit:
            df = df.head(row_limit)

        #Filtering out records with weeks of year into the future
        df = df[df['weekOfYear']< self._get_todays_week_no() + 1]

        #collapsing lpis over time to single list
        lpi_bw = df.groupby(['userId'])['lpi'].apply(lambda x: x.tolist())
        df = df.join(lpi_bw, how='inner', on='userId', lsuffix='_gv_id', rsuffix='_lpidf')

        #collapsing game variety over time to single list
        gv_bw = df.groupby(['userId'])['gameVariety'].apply(lambda x: x.tolist())
        df.join(gv_bw, how='inner', on='userId', lsuffix='_gv_id', rsuffix='_rightgv')

        #collapsing all weeks into single list
        gv_w = df.groupby(['userId'])['weekOfYear'].apply(lambda x: x.tolist())
        df.join(gv_w, how='inner', on='userId', lsuffix='_gv_id', rsuffix='_gv_w')

        self.set_gv_data(gv_df=df)
        return "LOGGING(FIX): GAME VAR DATA SET SUCCESSFULLY"

    def preprocess_subs_data(self, row_limit=None):
        df = load_data_by_key(key='subs_key',
                                bucket=self.get_s3_bucket(),
                                bucket_keys=self.get_aws_keys())
        df.reset_index(inplace=True)

        if row_limit:
            df = df.head(row_limit)

        df['date'] = pd.to_datetime(df['date'])
        #dummifying subcription data
        account_status = pd.get_dummies(df.state)
        pd.concat([df,account_status],axis=1)
        df.rename(columns={'subs':'is_subs_acct'})


        #Calculate the difference between every login by user from days of last login
        df['day_gap'] = df.groupby('user_id').date.diff()
        df['day_gap'] = df.day_gap.dt.days
        #creating churned user records
        df['churned'] = df['day_gap'].apply(lambda x: 1 if x > self.get_churn_threshold() else 0)
        churn_df = df.groupby('user_id')['churned'].max()
        df.join(churn_df, how="inner",on='user_id', lsuffix='_sub_df', rsuffix='_ch_df')

        ## identifying duplicate rows, because need 1 row for every user_id
        df['dup_row'] = df.sort_values(by=['user_id','churned']).duplicated(subset='user_id',keep='last')
        df = df[df['dup_row']==False]

        #adding count of how many times user has churned
        churn_counts_sorted = df.groupby('user_id').churned.sum().sort_values()
        df = df.join(churn_counts_sorted, how='inner', on='user_id', lsuffix='_subs', rsuffix='_churn_counts')
        df['prior_churn_count'] = df['churned_churn_counts']-df['churned_subs']

        df.workout_day.fillna(0,inplace=True)
        df.gameplay_day.fillna(0,inplace=True)
        df.day_gap.fillna(0,inplace=True)

        df.dropna(inplace=True)

        df.drop(['churned_churn_counts','index','dup_row','state','date','day_gap'],axis=1,inplace=True)
        df.rename(columns={'churned_subs':'churned'}, inplace=True)

        #dropping columns not used in model
        #df.drop(['churned_sub_df','level_0','index'],axis=1,inplace=True)

        self.set_subs_data(df)
        return "LOGGING(FIX): SUB DATA SET SUCCESSFULLY"

    def get_data_matrix(self):
        return self.get_subs_data().join(self.get_gid_data(), how='inner', on='user_id', lsuffix='_subs_df', rsuffix='_gmid_df')


    def preprocess_all_datasets(self, row_limit=None):
        # self.preprocess_demo_df(row_limit=row_limit)
        self.preprocess_game_id_data(row_limit=row_limit)
        # self.preprocess_game_variety_data(row_limit=row_limit)
        self.preprocess_subs_data(row_limit=row_limit)
        return "LOGGING(FIX): PREPROCESSED SUCCESSFULLY"

    def _get_todays_week_no(self):
        return date.today().isocalendar()[1]



if __name__ == "__main__":
    bucket = connect_bucket()
    aws_keys = get_keys_for_bucket()
    pipeline = ECPipeline()
    pipeline.set_s3_bucket(bucket)
    pipeline.set_aws_keys(aws_keys)
