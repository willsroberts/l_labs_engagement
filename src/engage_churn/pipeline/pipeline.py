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

    def preprocess_demo_df(self, row_limit=None):
        '''
        IN: Optional record limit
        DESC:
        1) Retrieves Data from AWS S3
        2) Dummifies and filters data set
        3) Function sets on pipeline object the cleaned demographic data
        OUT: N/A
        '''
        print "LOGGING (FIX): DEMO DATA SET PREPROCESSING"

        #
        #   RETRIEVE DATA
        #
        df_chunks = load_data_by_key(key='demo_key',
                              bucket=self.get_s3_bucket(),
                              bucket_keys=self.get_aws_keys(),
                              row_limit=row_limit)

        result_df = pd.DataFrame()

        for df in df_chunks:
            #
            #   DUMMIFY AND FILTER
            #
            # dummify gender
            df = self.dummify_sex_data(df)
            # No known individuals above age 117, so filtering those records
            # Unreasonable for age <5 for players
            df = self.filter_user_by_age(df)
            result_df = pd.concat((result_df,df), axis=0)

        # Dropping columns not needed
        result_df.drop(['f','gender','language','index'],axis=1,inplace=True)
        #
        #   SET RESULTS ONTO PIPELINE OBJ
        #
        self.set_demo_data(demo_df=result_df)
        return "LOGGING (FIX): DEMO DATA SET, SUCCESSFULLY"

    def dummify_sex_data(self, df):
        df_sex = pd.get_dummies(df.gender)
        df = pd.concat([df,df_sex],axis=1)
        df = df.rename(columns={'m':'is_male'})
        return df

    def filter_user_by_age(self, df):
        df = df[df['age'] < 117]
        df = df[df['age'] > 4]
        return df

    def preprocess_game_id_data(self, row_limit=None):
        '''
        IN: Optional record limit
        DESC:
        1)  Retrieves Data from AWS S3
        2)  Create features needed for last 5 lpis available to user
        3)  Create features needed for game play hour avg, std
        4)  Fill NaN Values with median
        OUT: N/A
        '''
        print "LOGGING(FIX): GAME ID DATA SET PREPROCESSING"

        #
        #   RETREIVE DATA
        #


        df_chunks = load_data_by_key(key='game_id_key',
                                bucket=self.get_s3_bucket(),
                                bucket_keys=self.get_aws_keys(),
                                row_limit=row_limit)

        result_df = pd.DataFrame()

        for df in df_chunks:
            df['date'] = pd.to_datetime(df.date)
            df = self.create_lpi_history_features(df)
            df = self.create_gaming_hours_features(df)
            df = self.create_game_id_features(df)
            df = self.fill_gid_na_vals(df)
            result_df = pd.concat((result_df,df), axis=0)


        result_df.rename(columns={'hour_gid':'session_hours',
                     'gameId_gid':'session_gameIds',
                     'hour_ugdf':'all_user_gaming_hours',
                     'gameId_ugpdf':'all_user_games_plyd'},inplace=True)
        result_df = self.drop_unused_gid_features(result_df)

        result_df.drop_duplicates(keep='last',inplace=True)
            #Setting dateframe on pipeline object
        self.set_gid_data(gid_df=result_df)
        return "LOGGING(FIX): GAME ID DATA SET SUCCESSFULLY"

    def fill_gid_na_vals(self, df):
        df.lpi.fillna(df.lpi.median(),                     inplace=True)
        df.ffth_last_lpi_change.fillna(df.ffth_last_lpi_change.median(),    inplace=True)
        df.frth_last_lpi_change.fillna(df.frth_last_lpi_change.median(),    inplace=True)
        df.thrd_last_lpi_change.fillna(df.thrd_last_lpi_change.median(),    inplace=True)
        df.sec_last_lpi_change.fillna(df.sec_last_lpi_change.median(),     inplace=True)
        df.last_lpi_change.fillna(df.last_lpi_change.median(),         inplace=True)
        return df

    def create_lpi_history_features(self, df):
        #Create Last 5 Week Change columns
        df ['delta_with_lw']        = df.groupby('user_id').lpi.diff()
        week_deltas = df.groupby('user_id')['delta_with_lw'].apply(lambda x: x.tolist())
        df = df.join(week_deltas, how='inner', on='user_id', lsuffix='_gvid', rsuffix='_wl')
        df['last_5_deltas']         = df.delta_with_lw_wl.apply(lambda x: x[-5:])
        # Last 5 Deltas
        delta_df  =  df.last_5_deltas.apply(lambda x: pd.Series(self._create_five_lpi_values(x)))
        delta_df.columns=['ffth_last_lpi_change','frth_last_lpi_change','thrd_last_lpi_change','sec_last_lpi_change','last_lpi_change']
        df = pd.concat([df,delta_df], axis=1)
        return df

    def create_gaming_hours_features(self, df):
        # Collect all user gaming hours
        q = df.groupby(['user_id'])['hour'].apply(lambda x: ','.join(x))
        df = df.join(q, on='user_id', how='inner', lsuffix='_gid', rsuffix='_ugdf')
        df.rename(columns={'hour_ugdf':'all_user_gaming_hours'}, inplace=True)
        # Convert strings of hours to integers
        list_of_ints = df.all_user_gaming_hours.apply(lambda x: map(int, x.split(',')))
        list_of_ints.name = 'game_hours_ints'
        df = pd.concat([df,list_of_ints],axis=1)

        df['hour_avg'] = df.game_hours_ints.apply(lambda x: np.mean(x))
        df['hour_std'] = df.game_hours_ints.apply(lambda x: np.std(x))
        return df

    def create_game_id_features(self, df):
        # Create column for all games played by userId
        # across all sessions
        r = df.groupby(['user_id'])['gameId'].apply(lambda x: ','.join(x))
        df = df.join(r, on='user_id', how='inner', lsuffix='_gid', rsuffix='_ugpdf')
        return df

    def drop_unused_gid_features(self,df):
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
        return df

    def preprocess_game_variety_data(self, row_limit=None):
        '''
        IN: Optional record limit
        OUT: N/A
        '''
        print "LOGGING(FIX): GAME VAR DATA SET PREPROCESSING"
        #if self.get_game_variety_data not none, prompt to make sure
        df_chunks = load_data_by_key(key='game_var_key',
                                bucket=self.get_s3_bucket(),
                                bucket_keys=self.get_aws_keys(),
                                row_limit=row_limit)

        result_df = pd.DataFrame()

        #Filtering out records with weeks of year into the future
        for df in df_chunks:
            df = self.create_game_variety_features(df)
            df = self.fill_gv_na_vals(df)
            df = self.drop_unused_gv_features(df)
            result_df = pd.concat((result_df,df), axis=0)

        result_df.drop_duplicates(keep='last',inplace=True)
        result_df.reset_index(drop=True,inplace=True)

        self.set_gv_data(gv_df=result_df)
        return "LOGGING(FIX): GAME VAR DATA SET SUCCESSFULLY"

    def create_game_variety_features(self, df):
        gv_bw = df.groupby(['user_id'])['gameVariety'].apply(lambda x: x.tolist())
        df = df.join(gv_bw, how='inner', on='user_id', lsuffix='_gv_id', rsuffix='_rightgv')
        df.rename(columns={'gameVariety_rightgv':'game_varieties'}, inplace=True)

        df['gv_max']        = df.game_varieties.apply(lambda x: np.max(x))
        df['gv_avg']        = df.game_varieties.apply(lambda x: np.mean(x))
        df['gv_std_dev']    = df.game_varieties.apply(lambda x: np.std(x))
        return df

    def fill_gv_na_vals(self,df):
        #collapsing all weeks into single list
        df.gv_max.fillna(0,inplace=True)
        df.gv_avg.fillna(0,inplace=True)
        df.gv_std_dev.fillna(0,inplace=True)
        return df

    def drop_unused_gv_features(self, df):
        df.drop(['index','weekOfYear','lpi', 'gameVariety_gv_id', 'game_varieties'],axis=1,inplace=True)
        return df

    def preprocess_subs_data(self, row_limit=None):
        '''
        IN: Optional record limit
        OUT: N/A
        '''
        print "LOGGING(FIX): SUB DATA SET PREPROCESSING"
        df_chunks = load_data_by_key(key='subs_key',
                                bucket=self.get_s3_bucket(),
                                bucket_keys=self.get_aws_keys(),
                                row_limit=row_limit)

        result_df = pd.DataFrame()

        for df in df_chunks:
            df['date'] = pd.to_datetime(df['date'])
            df = self.dummify_subs_account_data(df)
            df = self.create_churned_feature(df)
            df = self.create_churned_count_feature(df)
            df = self.fill_subs_na_vals(df)
            result_df = pd.concat((result_df,df), axis=0)

        result_df.drop(['churned_churn_counts','index','dup_row','state','date','day_gap'],axis=1,inplace=True)
        result_df.rename(columns={'churned_subs':'churned'}, inplace=True)

        self.set_subs_data(result_df)
        return "LOGGING(FIX): SUB DATA SET SUCCESSFULLY"

        #dummifying subcription data

    def dummify_subs_account_data(self, df):
        account_status = pd.get_dummies(df.state)
        pd.concat([df,account_status],axis=1)
        df.rename(columns={'subs':'is_subs_acct'})
        return df

    #Calculate the difference between every login by user from days of last login
    def create_churned_feature(self, df):
        df['day_gap'] = df.groupby('user_id').date.diff()
        df['day_gap'] = df.day_gap.dt.days
        #creating churned user records
        df['churned'] = df['day_gap'].apply(lambda x: 1 if x > self.get_churn_threshold() else 0)
        churn_df = df.groupby('user_id')['churned'].max()
        df.join(churn_df, how="inner",on='user_id', lsuffix='_sub_df', rsuffix='_ch_df')

        ## identifying duplicate rows, because need 1 row for every user_id
        df['dup_row'] = df.sort_values(by=['user_id','churned']).duplicated(subset='user_id',keep='last')
        df = df[df['dup_row']==False]
        return df

    #adding count of how many times user has churned
    def create_churned_count_feature(self, df):
        churn_counts_sorted = df.groupby('user_id').churned.sum().sort_values()
        df = df.join(churn_counts_sorted, how='inner', on='user_id', lsuffix='_subs', rsuffix='_churn_counts')
        df['prior_churn_count'] = df['churned_churn_counts']-df['churned_subs']
        return df

    def fill_subs_na_vals(self, df):
        df.workout_day.fillna(df.workout_day.median(),inplace=True)
        df.gameplay_day.fillna(df.gameplay_day.median(),inplace=True)
        df.day_gap.fillna(df.day_gap.median(),inplace=True)

        df.dropna(inplace=True)
        return df

    def get_data_matrix(self):
        '''
        IN:  N/A
        OUT: Result Data Matrix After Joining All Sub-components
        '''
        print "LOGGING(FIX): RETURNING DATA MATRIX"
        print "Uniq Users BEF JOIN 1 : subs:{} , gid:{}".format(pd.Series.nunique(self.get_subs_data().user_id), pd.Series.nunique(self.get_gid_data().user_id))
        d_mat = self.get_subs_data().join(self.get_gid_data(), how='inner', on='user_id', lsuffix='_subs_df', rsuffix='_gmid_df')
        # d_mat.reset_index(inplace=True)
        print d_mat.head(1)

        print "Uniq Users BEF JOIN 2 : mat:{} , gv:{}".format(pd.Series.nunique(d_mat.user_id), pd.Series.nunique(self.get_gv_data().user_id))
        d_mat = d_mat.join(self.get_gv_data(), how='inner', on='user_id', lsuffix='_tm_mat_df', rsuffix='_gvid_df')
        # d_mat.reset_index(inplace=True)
        print d_mat.head(1)

        print "Uniq Users BEF JOIN 3 : mat:{} , demo:{}".format(pd.Series.nunique(d_mat.user_id), pd.Series.nunique(self.get_demo_data().user_id))
        d_mat = d_mat.join(self.get_demo_data(), how='inner', on='user_id', lsuffix='_subs_gv_tm', rsuffix='_demo_df')
        # d_mat.reset_index(inplace=True)
        print d_mat.head(1)
        print "Uniq Users FINAL : {}".format(pd.Series.nunique(d_mat.user_id))

        d_mat.drop(['user_id_demo_df', 'user_id_gvid_df', 'user_id_subs_df', 'user_id_tm_mat_df', 'user_id_subs_gv_tm'], axis=1, inplace=True)

        return d_mat

    def preprocess_all_datasets(self, row_limit=None):
        '''
        IN:     Optional record limit
        DESC:   Parent function to call all preprocessing steps
        OUT:    N/A
        '''
        print "LOGGING(FIX): PREPROCESSING DATA"
        self.preprocess_demo_df(row_limit=row_limit)
        self.preprocess_game_id_data(row_limit=row_limit)
        self.preprocess_game_variety_data(row_limit=row_limit)
        self.preprocess_subs_data(row_limit=row_limit)
        return "LOGGING(FIX): PREPROCESSED SUCCESSFULLY"

    def _get_todays_week_no(self):
        return date.today().isocalendar()[1]

    def _create_five_lpi_values(self, x):
        vals = np.zeros(5)
        vals[:] = np.NAN
        for i in xrange(len(x)):
            vals[i] = x[i]
        return list(vals)

if __name__ == "__main__":
    bucket = connect_bucket()
    aws_keys = get_keys_for_bucket()
    pipeline = ECPipeline()
    pipeline.set_s3_bucket(bucket)
    pipeline.set_aws_keys(aws_keys)
