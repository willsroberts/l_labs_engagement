from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    r2_score,
    f1_score
    )
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from pipeline.pipeline import ECPipeline
from utilities.aws_bucket_client import display_all_keys, get_keys_for_bucket, load_data_by_key, connect_bucket
import grid_search_model

class EngagementModel(object):

    def __init__(self):
        self.gs_dict = grid_search_model.grid_search_dict()
        self.gs_results = None
        self.x_matrix = None
        self.labels = None
        self.optimal_model = None

    def set_gs_model_dict(self,gs_dict):
        self.gs_dict = gs_dict

    def get_gs_model_dict(self):
        return self.gs_dict

    def get_gs_results(self):
        return self.gs_results

    def set_gs_results(self, results):
        self.gs_results = results

    def set_x_matrix(self,x_mat):
        self.x_matrix = x_mat

    def get_x_matrix(self):
        return self.x_matrix

    def set_labels(self, labels):
        self.labels = labels

    def get_labels(self):
        return self.labels

    def set_optimal_model(self, model):
        self.optimal_model = model

    def get_optimal_model(self):
        return self.optimal_model

    def fit_grid_search(self, X_train, y_train):
        g_results = []
        for model, hyperparams in self.get_gs_model_dict():
            gs = GridSearchCV(model, hyperparams, n_jobs=1, scoring='recall') #roc_auc_score
            gs.fit(X_train, y_train)
            g_results.append(gs)
        self.set_gs_results(g_results)
        return "LOGGING (FIX): GRID SEARCH SET, SUCCESSFULLY"

    def report_model_results(self):
        for grid in self.get_gs_results():
            print "Grid Search Results: {}".format(grid.best_estimator_.__class__.__name__)
            print "    params: {}".format(grid.best_params_)
            print "    {} : {}".format(grid.get_params()['scoring'], grid.best_score_)

    def predict_and_score(self, models, X_test, y_test):
        '''
        returns the highest performing model from a list of models
        '''

        metrics = [precision_score, recall_score, accuracy_score, roc_auc_score, mean_squared_error]

        # prints all scores, but in the end we want to automatically select the highest scoring one
        for model in models:
            print model.__class__.__name__
            for metric in metrics:
                print "  ",metric.__name__, ":", metric(y_test, model.predict(X_test))
            print " "

        self.set_optimal_model(models[0])
        return "LOGGING (FIX): OPTIMAL MODEL SET, SUCCESSFULLY"

if __name__ == '__main__':
    pipeline = ECPipeline()
    pipeline.set_s3_bucket(connect_bucket())
    pipeline.set_aws_keys(get_keys_for_bucket())

    pipeline.preprocess_all_datasets()

    df = pipeline.get_data_matrix()
    print df.head()

    model = EngagementModel()
    model.set_labels(labels=df.pop('churned').values)
    model.set_x_matrix(x_mat=df.values)

    X_train, X_test, y_train, y_test = train_test_split(model.get_x_matrix(),
            model.get_labels(), test_size=0.20, random_state=42)

    model = LogisticRegression ()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)

    print model.coef_
    print "Model Score: {} , Precision Score: {}, Recall Score: {}".format(model.score(X_test, y_test),
           precision_score(y_test, y_predict),
           recall_score(y_test, y_predict))

    # model.fit_grid_search( X_train, y_train )
    #
    # model.report_model_results()
    #
    # all_models = [g.best_estimator_ for g in model.get_gs_results()]
    #
    # model.predict_and_score(all_models, X_test, y_test)
    #
    # model.predict_and_score()
