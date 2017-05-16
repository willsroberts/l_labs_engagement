import pandas as pd
import numpy as np
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

        df.drop(['f','gender','index','language'],axis=1,inplace=True)
        df.rename(columns={'userId':'user_id'}, inplace=True)

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

        #Create Last 3 Week Change columns

        df ['delta_with_lw']        = df.groupby('userId').lpi.diff()
        week_deltas = df.groupby('userId')['delta_with_lw'].apply(lambda x: x.tolist())
        df = df.join(week_deltas, how='inner', on='userId', lsuffix='_gvid', rsuffix='_wl')
        df['last_5_deltas']         = df.delta_with_lw_wl.apply(lambda x: x[-5:])
        # Last 5 Deltas
        # [1st, 2nd, 3rd, 4th, 5th]
        #
        # import ipdb; ipdb.set_trace()
        delta_df  =  df.last_5_deltas.apply(lambda x: pd.Series(self._create_five_lpi_values(x)))
        delta_df.columns=['ffth_last_lpi_change','frth_last_lpi_change','thrd_last_lpi_change','sec_last_lpi_change','last_lpi_change']
        df = pd.concat([df,delta_df], axis=1)

        # #Create column for all hours played by user
        # #across all sessions

        # Collect all user gaming hours
        q = df.groupby(['userId'])['hour'].apply(lambda x: ','.join(x))
        df = df.join(q, on='userId', how='inner', lsuffix='_gid', rsuffix='_ugdf')
        df.rename(columns={'hour_ugdf':'all_user_gaming_hours'}, inplace=True)
        # Convert strings of hours to integers
        list_of_ints = df.all_user_gaming_hours.apply(lambda x: self.return_list_from_string_col(x))
        list_of_ints.name = 'game_hours_ints'
        df = pd.concat([df,list_of_ints],axis=1)

        df['hour_avg'] = df.game_hours_ints.apply(lambda x: np.mean(x))
        df['hour_std'] = df.game_hours_ints.apply(lambda x: np.std(x))

        # Create column for all games played by userId
        # across all sessions
        r = df.groupby(['userId'])['gameId'].apply(lambda x: ','.join(x))
        df = df.join(r, on='userId', how='inner', lsuffix='_gid', rsuffix='_ugpdf')

        df.lpi.fillna(df.lpi.median(),                     inplace=True)
        df.ffth_last_lpi_change.fillna(df.ffth_last_lpi_change.median(),    inplace=True)
        df.frth_last_lpi_change.fillna(df.frth_last_lpi_change.median(),    inplace=True)
        df.thrd_last_lpi_change.fillna(df.thrd_last_lpi_change.median(),    inplace=True)
        df.sec_last_lpi_change.fillna(df.sec_last_lpi_change.median(),     inplace=True)
        df.last_lpi_change.fillna(df.last_lpi_change.median(),         inplace=True)


        df.rename(columns={'hour_gid':'session_hours',
                     'gameId_gid':'session_gameIds',
                     'hour_ugdf':'all_user_gaming_hours',
                     'gameId_ugpdf':'all_user_games_plyd',
                     'userId':'user_id'},inplace=True)


        df.drop(['date'],                   axis=1,inplace=True)
        df.drop(['index'],                  axis=1,inplace=True)
        df.drop(['last_5_deltas'] ,         axis=1,inplace=True)
        df.drop(['delta_with_lw_wl'],       axis=1,inplace=True)
        df.drop(['delta_with_lw_gvid'],     axis=1,inplace=True)
        df.drop(['all_user_gaming_hours'],  axis=1,inplace=True)
        df.drop(['all_user_games_plyd'],    axis=1,inplace=True)
        df.drop(['session_gameIds'],        axis=1,inplace=True)
        df.drop(['session_hours'],          axis=1,inplace=True)
        df.drop(['game_hours_ints'],        axis=1,inplace=True)
        df.drop(['lpi'],                    axis=1,inplace=True)

        df.drop_duplicates(keep='last',inplace=True)

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

        #setting game variety metrics - max, std, avg
        df.rename(columns={'userId':'user_id'},inplace=True)

        gv_bw = df.groupby(['user_id'])['gameVariety'].apply(lambda x: x.tolist())
        df = df.join(gv_bw, how='inner', on='user_id', lsuffix='_gv_id', rsuffix='_rightgv')
        df.rename(columns={'gameVariety_rightgv':'game_varieties'}, inplace=True)

        df['gv_max']        = df.game_varieties.apply(lambda x: np.max(x))
        df['gv_avg']        = df.game_varieties.apply(lambda x: np.mean(x))
        df['gv_std_dev']    = df.game_varieties.apply(lambda x: np.std(x))

        #collapsing all weeks into single list
        df.gv_max.fillna(0,inplace=True)
        df.gv_avg.fillna(0,inplace=True)
        df.gv_std_dev.fillna(0,inplace=True)

        df.drop(['index','weekOfYear','lpi', 'gameVariety_gv_id', 'game_varieties'],axis=1,inplace=True)

        df.drop_duplicates(keep='last', inplace=True)

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

        df.workout_day.fillna(df.workout_day.median(),inplace=True)
        df.gameplay_day.fillna(df.gameplay_day.median(),inplace=True)
        df.day_gap.fillna(df.day_gap.median(),inplace=True)

        df.dropna(inplace=True)

        df.drop(['churned_churn_counts','index','dup_row','state','date','day_gap'],axis=1,inplace=True)
        df.rename(columns={'churned_subs':'churned'}, inplace=True)

        #dropping columns not used in model
        #df.drop(['churned_sub_df','level_0','index'],axis=1,inplace=True)

        self.set_subs_data(df)
        return "LOGGING(FIX): SUB DATA SET SUCCESSFULLY"

    def get_data_matrix(self):
        print "LOGGING(FIX): RETURNING DATA MATRIX"
        d_mat = self.get_subs_data().join(self.get_gid_data(), how='inner', on='user_id', lsuffix='_subs_df', rsuffix='_gmid_df')
        print d_mat.head()
        d_mat = d_mat.join(self.get_gv_data(), how='inner', on='user_id', lsuffix='_tm_mat_df', rsuffix='_gvid_df')
        print d_mat.head()
        d_mat = d_mat.join(self.get_demo_data(), how='inner', on='user_id', lsuffix='_subs_gv_tm', rsuffix='_demo_df')
        print d_mat.head()

        return d_mat


    def preprocess_all_datasets(self, row_limit=None):
        self.preprocess_demo_df(row_limit=row_limit)
        self.preprocess_game_id_data(row_limit=row_limit)
        self.preprocess_game_variety_data(row_limit=row_limit)
        self.preprocess_subs_data(row_limit=row_limit)
        return "LOGGING(FIX): PREPROCESSED SUCCESSFULLY"

    def _get_todays_week_no(self):
        return date.today().isocalendar()[1]

    def _create_five_lpi_values(self, x):
        # import ipdb; ipdb.set_trace()
        vals = np.zeros(5)
        vals[:] = np.NAN
        for i in xrange(len(x)):
            vals[i] = x[i]
        return list(vals)

    def return_list_from_string_col(self, x):
        # import ipdb; ipdb.set_trace()
        return [int(y) for y in x.split(',')]

    def return_mean_of_list(self,x):
        return np.mean(x)

if __name__ == "__main__":
    bucket = connect_bucket()
    aws_keys = get_keys_for_bucket()
    pipeline = ECPipeline()
    pipeline.set_s3_bucket(bucket)
    pipeline.set_aws_keys(aws_keys)
