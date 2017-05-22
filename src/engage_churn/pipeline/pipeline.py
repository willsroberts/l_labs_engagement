import pandas as pd
import numpy as np
import os, gc
import logging
import pprint
from datetime import timedelta, date
from collections import Counter
from time import sleep
# from utilties.logging import setup_logging
from utilities.aws_bucket_client import (
    display_all_keys,
    get_keys_for_bucket,
    load_data_by_key,
    connect_bucket,
    write_intermed_data_to_s3
)

class ECPipeline(object):

    logging.basicConfig(filename='../logs/apps.log', level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    def __init__(self, bucket=None, aws_keys=None, write_intermeds=True):
        self.bucket = bucket
        self.aws_keys = None
        self.demo_data = None
        self.gv_data = None
        self.gid_data = None
        self.subs_data = None
        self.churn_threshold = 15
        self.write_intermeds = write_intermeds

    def get_s3_bucket(self):
        return self.bucket

    def set_s3_bucket(self, bucket):
        self.bucket = bucket

    def get_aws_keys(self):
        return self.aws_keys

    def set_aws_keys(self, keys):
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

    def set_gv_data(self, gv_df):
        self.gv_data = gv_df

    def get_gid_data(self):
        '''
        WARN: BE CAUTIOUS RETRIEVING LARGE DS
        '''
        return self.gid_data

    def set_gid_data(self, gid_df):
        self.gid_data = gid_df

    def get_churn_threshold(self):
        return self.churn_threshold

    def set_churn_threshold(self, churn_thresh):
        self.churn_threshold = churn_thresh

    def get_subs_data(self):
        return self.subs_data

    def set_subs_data(self, subs_df):
        self.subs_data = subs_df

    def set_write_intermeds(self, write_intermeds):
        self.write_intermeds = write_intermeds

    def get_write_intermeds(self):
        return self.write_intermeds

#####################################################

    def preprocess_demo_data(self, row_limit=None):
        '''
        IN: Optional record limit
        DESC:
        1) Retrieves Data from AWS S3
        2) Dummifies and filters data set
        3) Function sets on pipeline object the cleaned demographic data
        OUT: N/A
        '''
        logging.info('DEMO_DATA - SET PREPROCESSING')

        #
        #   RETRIEVE DATA
        #
        df_chunks = load_data_by_key(key='demo_key',
                                     bucket=self.get_s3_bucket(),
                                     bucket_keys=self.get_aws_keys(),
                                     row_limit=row_limit)

        result_df = pd.DataFrame()

        logging.info('DEMO_DATA - CHUNKS LOOP START')
        for df in df_chunks:
            #
            #   DUMMIFY AND FILTER
            #
            # dummify gender
            df = self.dummify_sex_data(df)
            # No known individuals above age 117, so filtering those records
            # Unreasonable for age <5 for players
            df = self.filter_user_by_age(df)
            result_df = pd.concat((result_df, df), axis=0)
            del df
        logging.info('DEMO_DATA - CHUNKS LOOP END')

        del df_chunks
        gc.collect()

        # Dropping columns not needed
        result_df.drop(['f', 'gender', 'language', 'index'], axis=1, inplace=True)
        logging.info('DEMO_DATA - dropping columns')
        #
        #   SET RESULTS ONTO PIPELINE OBJ
        #
        result_df.reset_index(drop=True, inplace=True)
        self.set_demo_data(demo_df=result_df)
        if self.get_write_intermeds():
            logging.info('DEMO_DATA - WRITE INTERMEDIATE')
            write_intermed_data_to_s3(bucket=self.get_s3_bucket(),
                                  bucket_keys=self.get_aws_keys(),
                                  df=result_df,
                                  file_name='intermed_demograph_',
                                  H=False)
        logging.info('DEMO_DATA - PREPROCESSING COMPLETE')
        return None

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
        logging.info('GAMEID_DATA - PREPROCESSING')

        #
        #   RETREIVE DATA
        #

        df_chunks = load_data_by_key(key='game_id_key',
                                     bucket=self.get_s3_bucket(),
                                     bucket_keys=self.get_aws_keys(),
                                     row_limit=row_limit)

        result_df = pd.DataFrame()
        logging.info('GAMEID_DATA - CHUNKS LOOP START')
        for df in df_chunks:
            df['date'] = pd.to_datetime(df.date)
            df = self.create_feats_lpi_history(df)
            df = self.create_feats_gaming_hours(df)
            df = self.create_feat_tot_usr_games(df)
            df = self.fill_gid_na_vals(df)
            result_df = pd.concat((result_df, df), axis=0)
            del df
        logging.info('GAMEID_DATA - CHUNKS LOOP END')
        del df_chunks
        gc.collect()
        logging.info('GAMEID_DATA - RENAMING COLUMN')
        result_df.rename(columns={'hour_gid': 'session_hours',
                                  'gameId_gid': 'session_gameIds',
                                  'hour_ugdf': 'all_user_gaming_hours',
                                  'gameId_ugpdf': 'all_user_games_plyd',
                                  'lpi_flpi':'first_recorded_lpi'}, inplace=True)
        result_df = self.drop_unused_gid_features(result_df)

        #Dropping users that have no lpi recorded at all
        logging.info('GAMEID_DATA - DROPPING RECORDS NULL LPI')
        result_df = result_df[result_df['first_recorded_lpi'].notnull()]

        result_df.drop_duplicates(keep='last', inplace=True)
        result_df.reset_index(drop=True, inplace=True)
        self.set_gid_data(gid_df=result_df)

        if self.get_write_intermeds():
            logging.info('GAMEID_DATA - WRITE INTERMEDIATE')
            write_intermed_data_to_s3(bucket=self.get_s3_bucket(),
                                  bucket_keys=self.get_aws_keys(),
                                  df=result_df,
                                  file_name='intermed_gameid_',
                                  H=False)
        logging.info('GAMEID_DATA - PREPROCESSING COMPLETE')
        return None

    def preprocess_game_variety_data(self, row_limit=None):
        '''
        IN: Optional record limit
        OUT: N/A
        '''
        logging.info('GAMEVAR_DATA - PREPROCESSING')
        # if self.get_game_variety_data not none, prompt to make sure
        df_chunks = load_data_by_key(key='game_var_key',
                                     bucket=self.get_s3_bucket(),
                                     bucket_keys=self.get_aws_keys(),
                                     row_limit=row_limit)

        result_df = pd.DataFrame()

        # Filtering out records with weeks of year into the future
        logging.info('GAMEVAR_DATA - CHUNK LOOP START')
        for df in df_chunks:
            df = self.create_feats_game_variety(df)
            df = self.fill_gv_na_vals(df)
            df = self.drop_unused_gv_features(df)
            result_df = pd.concat((result_df, df), axis=0)
            del df
        logging.info('GAMEVAR_DATA - CHUNK LOOP END')

        del df_chunks
        gc.collect()

        result_df.drop_duplicates(keep='last', inplace=True)
        result_df.reset_index(drop=True, inplace=True)

        self.set_gv_data(gv_df=result_df)
        logging.info('')
        if self.get_write_intermeds():
            write_intermed_data_to_s3(bucket=self.get_s3_bucket(),
                                  bucket_keys=self.get_aws_keys(),
                                  df=result_df,
                                  file_name='intermed_gamevar_',
                                  H=False)
        logging.info('GAMEVAR_DATA - PREPROCESSING COMPLETE')
        return None

    def preprocess_subs_data(self, row_limit=None):
        '''
        IN: Optional record limit
        OUT: N/A
        '''
        logging.info('SUB_DATA - PREPROCESSING')
        df_chunks = load_data_by_key(key='subs_key',
                                     bucket=self.get_s3_bucket(),
                                     bucket_keys=self.get_aws_keys(),
                                     row_limit=row_limit)

        result_df = pd.DataFrame()
        logging.info('SUB_DATA - CHUNK LOOP START')
        for df in df_chunks:
            df['date'] = pd.to_datetime(df['date'])
            df = self.dummify_subs_account_data(df)
            df = self.create_feat_day_gap(df)
            df = self.create_feat_churned(df)
            df = self.create_feat_churned_count(df)
            df = self.fill_subs_na_vals(df)
            result_df = pd.concat((result_df, df), axis=0)
            del df
        logging.info('SUB_DATA - CHUNK LOOP END')

        del df_chunks
        gc.collect()

        result_df.drop(['churned_churn_counts', 'index', 'dup_row', 'state', 'date', 'day_gap'], axis=1, inplace=True)
        result_df.rename(columns={'churned_subs': 'churned'}, inplace=True)
        result_df.reset_index(drop=True, inplace=True)
        self.set_subs_data(result_df)
        if self.get_write_intermeds():
            logging.info('SUB_DATA - WRITE INTERMEDIATE')
            write_intermed_data_to_s3(bucket=self.get_s3_bucket(),
                                  bucket_keys=self.get_aws_keys(),
                                  df=result_df,
                                  file_name='intermed_subs_',
                                  H=False)
        logging.info('SUB_DATA - PREPROCESSING COMPLETE')
        return None

    def dummify_sex_data(self, df):
        logging.info('  demo - dummify sex data: start')
        df_sex = pd.get_dummies(df.gender)
        df = pd.concat([df, df_sex], axis=1)
        df = df.rename(columns={'m': 'is_male'})
        logging.info('  demo - dummify sex data: end')
        return df

    def filter_user_by_age(self, df):
        logging.info('  demo - filter by user age: start')
        df = df[df['age'] < 117]
        df = df[df['age'] > 4]
        logging.info('  demo - filter by user age: end')
        return df

    def fill_gid_na_vals(self, df):
        logging.info('  game_id - fill na values: start')
        df.lpi.fillna(df.lpi.median(), inplace=True)
        df.ffth_last_lpi_change.fillna(0, inplace=True)
        df.frth_last_lpi_change.fillna(0, inplace=True)
        df.thrd_last_lpi_change.fillna(0, inplace=True)
        df.sec_last_lpi_change.fillna(0, inplace=True)
        df.last_lpi_change.fillna(0, inplace=True)
        logging.info('  game_id - fill na values: end')
        return df

    def create_feat_first_lpi(self, df):
        logging.info('      game_id - create first lpi: start')
        #create first lpi record
        a = df.groupby('user_id')['lpi'].first()
        df = df.join(a, how='inner', on='user_id', lsuffix='_og', rsuffix='_flpi')
        del a
        df.rename(columns={'lpi_og':'lpi'},inplace=True)
        gc.collect()
        logging.info('      game_id - create first lpi: end')
        return df

    def create_feat_ultimate_five_lpi_diffs_list(self, df):
        logging.info('      game_id - create last five lpis list: start')
        df['delta_with_lw'] = df.groupby('user_id').lpi.diff()
        w = df.groupby('user_id')['delta_with_lw'].apply(lambda x: x.tolist())
        df = df.join(w, how='inner', on='user_id', lsuffix='_gvid', rsuffix='_wl')
        del w
        gc.collect()
        df['last_5_deltas'] = df.delta_with_lw_wl.apply(lambda x: x[-5:])
        logging.info('      game_id - create last five lpis list: end')
        return df

    def create_feats_ultimate_five_lpis(self, df):
        logging.info('      game_id - create last five lpis deltas: start')
        d = df.last_5_deltas.apply(lambda x: pd.Series(self._create_five_lpi_values(x)))
        d.columns = ['ffth_last_lpi_change', 'frth_last_lpi_change', 'thrd_last_lpi_change', 'sec_last_lpi_change', 'last_lpi_change']
        df = pd.concat([df, d], axis=1)
        del d
        gc.collect()
        logging.info('      game_id - create last five lpis deltas: end')
        return df

    def create_feats_lpi_history(self, df):
        logging.info('  game_id - create last lpi features: start')
        df = self.create_feat_first_lpi(df)
        df = self.create_feat_ultimate_five_lpi_diffs_list(df)
        df = self.create_feats_ultimate_five_lpis(df)
        logging.info('  game_id - create last lpi features: end')
        return df

    def create_feat_avg_gameplay_length(self, df):
        logging.info('      game_id - create average gameplay length features: start')
        df['session_hour_play'] = df.hour.apply(lambda x: map(int, x.split(',')))
        df['length_of_session'] = df.session_hour_play.apply(lambda x: abs(x[len(x)-1] - x[0]))
        p = df.groupby('user_id')['length_of_session'].mean()
        df = df.join(p, on='user_id', how='inner', lsuffix='_hrdf', rsuffix='_avgs')
        del p
        gc.collect()
        logging.info('      game_id - create average gameplay length features: start')
        return df

    def create_feats_avg_std_gameplay_hour(self, df):
        logging.info('      game_id - create average/std dev gameplay hours features: start')
        q = df.groupby('user_id')['hour'].apply(lambda x: map(int, (",".join(x)).split(',')))
        df = df.join(q, on='user_id', how='inner', lsuffix='_gid', rsuffix='_ugdf')
        df.rename(columns={'hour_ugdf': 'all_user_gaming_hours'}, inplace=True)
        del q
        gc.collect()
        df['hour_avg'] = df.all_user_gaming_hours.apply(lambda x: np.mean(x))
        df['hour_std'] = df.all_user_gaming_hours.apply(lambda x: np.std(x))
        logging.info('      game_id - create average/std dev gameplay hours features: end')
        return df

    def create_feats_gaming_hours(self, df):
        '''
        '''
        logging.info('  game_id - create gaming hours features: start')
        df = self.create_feat_avg_gameplay_length(df)
        df = self.create_feats_avg_std_gameplay_hour(df)
        logging.info('  game_id - create gaming hours features: end')
        return df

    def create_feat_tot_usr_games(self, df):
        # Create column for all games played by userId
        # across all sessions
        logging.info('  game_id - create total user features: start')
        r = df.groupby(['user_id'])['gameId'].apply(lambda x: ','.join(x))
        df = df.join(r, on='user_id', how='inner', lsuffix='_gid', rsuffix='_ugpdf')
        del r
        gc.collect()
        logging.info('  game_id - create total user features: start')
        return df

    def drop_unused_gid_features(self, df):
        logging.info('  game_id - dropping unused columns: start')
        df.drop(['date'], axis=1, inplace=True)
        df.drop(['index'], axis=1, inplace=True)
        df.drop(['last_5_deltas'], axis=1, inplace=True)
        df.drop(['delta_with_lw_wl'], axis=1, inplace=True)
        df.drop(['delta_with_lw_gvid'], axis=1, inplace=True)
        df.drop(['all_user_gaming_hours'], axis=1, inplace=True)
        df.drop(['all_user_games_plyd'], axis=1, inplace=True)
        df.drop(['session_gameIds'], axis=1, inplace=True)
        df.drop(['session_hours'], axis=1, inplace=True)
        df.drop(['lpi'], axis=1, inplace=True)
        df.drop(['length_of_session_hrdf'], axis=1, inplace=True)
        df.drop(['session_hour_play'],axis=1, inplace=True)
        gc.collect()
        logging.info('  game_id - dropping unused columns: end')
        return df

    def create_feats_game_variety(self, df):
        logging.info('  game_var - creating features for game variety: start')
        p = df.groupby(['user_id'])['gameVariety'].apply(lambda x: x.tolist())
        df = df.join(p, how='inner', on='user_id', lsuffix='_gv_id', rsuffix='_rightgv')
        df.rename(columns={'gameVariety_rightgv': 'game_varieties'}, inplace=True)
        del p
        gc.collect()
        logging.info('  game_var - max, mean, standard deviation')
        df['gv_max'] = df.game_varieties.apply(lambda x: np.max(x))
        df['gv_avg'] = df.game_varieties.apply(lambda x: np.mean(x))
        df['gv_std_dev'] = df.game_varieties.apply(lambda x: np.std(x))
        logging.info('  game_var - creating features for game variety: end')
        return df

    def fill_gv_na_vals(self, df):
        # collapsing all weeks into single list
        logging.info('  game_var - dropping na values: start')
        df.gv_max.fillna(0, inplace=True)
        df.gv_avg.fillna(0, inplace=True)
        df.gv_std_dev.fillna(0, inplace=True)
        logging.info('  game_var - dropping na values: end')
        return df

    def drop_unused_gv_features(self, df):
        df.drop(['index', 'weekOfYear', 'lpi', 'gameVariety_gv_id', 'game_varieties'], axis=1, inplace=True)
        return df
        # dummifying subcription data

    def dummify_subs_account_data(self, df):
        logging.info('  subs - dropping na values: start')
        account_status = pd.get_dummies(df.state)
        pd.concat([df, account_status], axis=1)
        df.rename(columns={'subs': 'is_subs_acct'})
        logging.info('  subs - dropping na values: end')
        return df

    def create_feat_day_gap(self,df):
        logging.info('  subs - create day gap: start')
        df['day_gap'] = df.groupby('user_id').date.diff()
        df['day_gap'] = df.day_gap.dt.days
        logging.info('  subs - create day gap: start')
        return df

    def drop_dup_churned(self, df):
        # identifying duplicate rows, because need 1 row for every user_id
        logging.info('  subs - dropping duplicate churned: start')
        df['dup_row'] = df.sort_values(by=['user_id', 'churned']).duplicated(subset='user_id', keep='last')
        df = df[df['dup_row'] == False]
        logging.info('  subs - dropping duplicate churned: end')
        return df

    def create_feat_churned(self, df):
        logging.info('  subs - creating churned feature: start')
        df['day_gap'] = df.groupby('user_id').date.diff()
        df['day_gap'] = df.day_gap.dt.days
        # creating churned user records
        df['churned'] = df['day_gap'].apply(lambda x: 1 if x > self.get_churn_threshold() else 0)
        c = df.groupby('user_id')['churned'].max()
        df.join(c, how="inner", on='user_id', lsuffix='_sub_df', rsuffix='_ch_df')
        del c

        # identifying duplicate rows, because need 1 row for every user_id
        df['dup_row'] = df.sort_values(by=['user_id', 'churned']).duplicated(subset='user_id', keep='last')
        df = df[df['dup_row'] == False]
        logging.info('  subs - creating churned feature: end')
        return df

    # adding count of how many times user has churned
    def create_feat_churned_count(self, df):
        logging.info('  subs - creating churned count feature: start')
        c = df.groupby('user_id').churned.sum().sort_values()
        df = df.join(c, how='inner', on='user_id', lsuffix='_subs', rsuffix='_churn_counts')
        df['prior_churn_count'] = df['churned_churn_counts'] - df['churned_subs']
        del c
        gc.collect()
        logging.info('  subs - creating churned count feature: end')
        return df

    def fill_subs_na_vals(self, df):
        logging.info('  subs - filling subs data: start')
        df.workout_day.fillna(df.workout_day.median(), inplace=True)
        df.gameplay_day.fillna(df.gameplay_day.median(), inplace=True)
        df.day_gap.fillna(df.day_gap.median(), inplace=True)
        df.dropna(inplace=True)
        logging.info('  subs - filling subs data: end')
        return df

    def create_feat_counter_mat_of_user_games(self, df):
        logging.info('  clustering - gameplay counter: start')
        df = df.groupby('user_id').gameId.apply(lambda x: Counter(map(int, ",".join(x.tolist()).split(','))))
        new_df = pd.DataFrame
        new_df = df.reset_index()
        new_df.rename(columns={'level_1':'game_id', 'gameID':'game_count'},inplace=True)
        logging.info('  clustering - gameplay counter: end')
        return new_df

    def create_feat_proportions_from_gameplay_counter(self, df):
        logging.info('  clustering - gameplay counter - sum: start')
        usrgmtotal = df.groupby('user_id').gameId.apply(lambda x: sum(Counter(map(int, ",".join(x.tolist()).split(','))).values()))
        df = df.join(usrgmtotal, on='userId', how='inner', lsuffix='_eachgame', rsuffix='tot')
        df['game_proportion'] = df['gameId_eachgame']/df['gameIdtot']
        knn_mat = df.drop(['gameId_eachgame','gameIdtot'])
        del df, usrgmtotal
        gc.collect()
        logging.info('  clustering - gameplay counter - sum: end')
        return knn_mat

    def get_knn_matrix(self):
        logging.info('Clustering - Building KNN Matrix: start')
        df = pd.read_csv(pipeline.get_s3_bucket().get_key(pipeline.get_aws_keys()[str('game_id_key')]))
        df.rename(columns={'userId':'user_id'},inplace=True)
        a = df.groupby('user_id').gameId.apply(lambda x: Counter(map(int, ",".join(x.tolist()).split(','))))
        b = pd.DataFrame
        b = a.reset_index()
        b.rename(columns={'level_1':'game_id', 'gameID':'game_count'},inplace=True)
        c = df.groupby('user_id').gameId.apply(lambda x: sum(Counter(map(int, ",".join(x.tolist()).split(','))).values()))
        df = b.join(c, on='user_id', how='inner', lsuffix='_eachgame', rsuffix='tot')
        df['game_proportion'] = df['gameId_eachgame']/df['gameIdtot']
        knn_mat = df.drop(['gameId_eachgame','gameIdtot'], axis=1)
        del b, c
        gc.collect()
        knn_mat = df.pivot_table(columns='game_id', index='user_id').fillna(0)
        if self.get_write_intermeds():
            write_intermed_data_to_s3(bucket=self.get_s3_bucket(),
                                  bucket_keys=self.get_aws_keys(),
                                  df=d_mat,
                                  file_name='final_matrix_',
                                  H=False)

        logging.info('Clustering - Building KNN Matrix: end')
        return knn_mat.values

    def get_data_matrix(self):
        '''
        IN:  N/A
        OUT: Result Data Matrix After Joining All Sub-components
        '''
        logging.info('Final Processing - DATA MATRIX: start')
        logging.info('Uniq Users BEF JOIN 1 : subs:{} , gid:{}'.format(pd.Series.nunique(self.get_subs_data().user_id), pd.Series.nunique(self.get_gid_data().user_id)))
        logging.info('Original Columns => subs: {}, gid: {}'.format(self.get_subs_data().columns, self.get_gid_data().columns))
        d_mat = self.get_subs_data().join(self.get_gid_data(), how='inner', on='user_id', lsuffix='_subs_df', rsuffix='_gmid_df')
        d_mat.reset_index(drop=True, inplace=True)

        logging.info('Uniq Users BEF JOIN 2 : mat:{} , gv:{}, \n columns: {}'.format(pd.Series.nunique(d_mat.user_id), pd.Series.nunique(self.get_gv_data().user_id), d_mat.columns))
        d_mat = d_mat.join(self.get_gv_data(), how='inner', on='user_id', lsuffix='_tm_mat_df', rsuffix='_gvid_df')
        d_mat.reset_index(drop=True, inplace=True)

        logging.info('Uniq Users BEF JOIN 3 : mat:{} , demo:{}, \n columns: {}'.format(pd.Series.nunique(d_mat.user_id), pd.Series.nunique(self.get_demo_data().user_id), d_mat.columns))
        d_mat = d_mat.join(self.get_demo_data(), how='inner', on='user_id', lsuffix='_subs_gv_tm', rsuffix='_demo_df')
        d_mat.reset_index(drop=True, inplace=True)
        logging.info('Uniq Users FINAL : {}'.format(pd.Series.nunique(d_mat.user_id)))
        d_mat.drop(['user_id_demo_df', 'user_id_gvid_df', 'user_id_subs_df', 'user_id_tm_mat_df', 'user_id_subs_gv_tm', 'user_id', 'user_id_gmid_df'], axis=1, inplace=True)
        logging.info('Final Matrix Columns: \n  {}'.format(d_mat.columns))

        if self.get_write_intermeds():
            write_intermed_data_to_s3(bucket=self.get_s3_bucket(),
                                  bucket_keys=self.get_aws_keys(),
                                  df=d_mat,
                                  file_name='knn_matrix_',
                                  H=False)

        logging.info('Final Processing - DATA MATRIX: end')
        return d_mat

    def preprocess_all_datasets(self, row_limit=None):
        '''
        IN:     Optional record limit
        DESC:   Parent function to call all preprocessing steps
        OUT:    N/A
        '''
        logging.info('EC PIPELINE - PREPROCESSING DATA: start')
        self.preprocess_demo_data(row_limit=row_limit)
        self.preprocess_game_id_data(row_limit=row_limit)
        self.preprocess_game_variety_data(row_limit=row_limit)
        self.preprocess_subs_data(row_limit=row_limit)
        logging.info('EC PIPELINE - PREPROCESSING DATA: end')
        return None

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
