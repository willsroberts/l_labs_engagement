from pipeline.pipeline import ECPipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
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
from timeit import default_timer as timer
from utilities.aws_bucket_client import display_all_keys, get_keys_for_bucket, load_data_by_key, connect_bucket
import grid_search_model

import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from itertools import combinations, izip
import random
import numpy as np
from scipy.spatial.distance import euclidean
from collections import defaultdict


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
                y_predict = model.predict(X_test)
                print "  ",metric.__name__, ":", metric(y_test, y_predict )
                # print classification_report(y_test, y_predict)
            print " "


        self.set_optimal_model(models[0])
        return "LOGGING (FIX): OPTIMAL MODEL SET, SUCCESSFULLY"

    def logistic_regression_report(self, X_train, y_train, X_test, y_test):
        m = LogisticRegression()
        m.fit(X_train, y_train)
        y_predict = m.predict(X_test)

        print model.coef_
        print "Model Score: {} , Precision Score: {}, Recall Score: {}".format(m.score(X_test, y_test),
               precision_score(y_test, y_predict),
               recall_score(y_test, y_predict))



    def cluster_users_by_game(self, pipeline):
        X = pipeline.get_knn_matrix()
        plot_k_sse(X, 2, 10)
        plot_k_silhouette(X, 2, 10)
        plot_all_2d(X, np.array(iris.feature_names), k=5)

    def k_means(X, k=5, max_iter=1000):
        """Performs k means

        Args:
        - X - feature matrix
        - k - number of clusters
        - max_iter - maximum iteratations

        Returns:
        - clusters - dict mapping cluster centers to observations
        """
        centers = [tuple(pt) for pt in random.sample(X, k)]
        for i in xrange(max_iter):
            clusters = defaultdict(list)

            for datapoint in X:
                distances = [euclidean(datapoint, center) for center in centers]
                center = centers[np.argmin(distances)]
                clusters[center].append(datapoint)

            new_centers = []
            for center, pts in clusters.iteritems():
                new_center = np.mean(pts, axis=0)
                new_centers.append(tuple(new_center))

            if set(new_centers) == set(centers):
                break

            centers = new_centers

        return clusters


    def sse(clusters):
        """Sum squared euclidean distance of all points to their cluster center"""
        sum_squared_residuals = 0
        for center, pts in clusters.iteritems():
            for pt in pts:
                sum_squared_residuals += euclidean(pt, center)**2
        return sum_squared_residuals


    def plot_k_sse(X, min_k, max_k):
        """Plots sse for values of k between min_k and max_k

        Args:
        - X - feature matrix
        - min_k, max_k - smallest and largest k to plot sse for
        """
        k_values = range(min_k, max_k+1)
        sse_values = []
        for k in k_values:
            clusters = k_means(X, k=k)
            sse_values.append(sse(clusters))
        plt.plot(k_values, sse_values)
        plt.xlabel('k')
        plt.ylabel('sum squared error')
        plt.show()


    def turn_clusters_into_labels(clusters):
        """Converts clusters dict returned by k_means into X, y (labels)

        Args:
        - clusters - dict mapping cluster centers to observations
        """
        labels = []
        new_X = []
        label = 0
        for cluster, pts in clusters.iteritems():
            for pt in pts:
                new_X.append(pt)
                labels.append(label)
            label += 1
        return np.array(new_X), np.array(labels)


    def plot_k_silhouette(X, min_k, max_k):
        """Plots sse for values of k between min_k and max_k

        Args:
        - X - feature matrix
        - min_k, max_k - smallest and largest k to plot sse for
        """
        k_values = range(min_k, max_k+1)
        silhouette_scores = []
        for k in k_values:
            clusters = k_means(X, k=k)
            new_X, labels = turn_clusters_into_labels(clusters)
            silhouette_scores.append(silhouette_score(new_X, labels))

        plt.plot(k_values, silhouette_scores)
        plt.xlabel('k')
        plt.ylabel('silhouette score')
        plt.show()


    def plot_all_2d(X, feature_names, k=3):
        """Generates all possible 2d plots of observations color coded by cluster ID"""
        pairs = list(combinations(range(X.shape[1]), 2))
        fig, axes = plt.subplots((len(pairs) / 2), 2)
        flattened_axes = [ax for ls in axes for ax in ls]

        for pair, ax in izip(pairs, flattened_axes):
            pair = np.array(pair)
            plot_data_2d(X[:, pair], feature_names[pair], ax, k=k)
        plt.show()


    def plot_data_2d(X, plot_labels, ax, k=3):
        """Generates single 2d plot of observations color coded by cluster ID"""
        clusters = k_means(X, k=k)
        new_X, labels = turn_clusters_into_labels(clusters)
        ax.scatter(new_X[:, 0], new_X[:, 1], c=labels)
        ax.set_xlabel(plot_labels[0])
        ax.set_ylabel(plot_labels[1])



if __name__ == '__main__':
    #Instantiate the data pipeline and preprocess_all_datasets
    #Option to limit sample size you're running on
    pipeline = ECPipeline(write_intermeds=True)
    pipeline.set_s3_bucket(connect_bucket())
    pipeline.set_aws_keys(get_keys_for_bucket())
    pipeline.preprocess_all_datasets()

    #Returns matrix to run predictions from in pipeline
    df = pipeline.get_data_matrix()

    #Instantiate Model
    model = EngagementModel()
    model.set_labels(labels=df.pop('churned').values)
    model.set_x_matrix(x_mat=df.values)

    model.cluster_users_by_game()

    #Create Test-Train Split - coercing a 20% label presence in sample
    start = timer()
    X_train, X_test, y_train, y_test = train_test_split(model.get_x_matrix(),
            model.get_labels(), test_size=0.20, random_state=42, stratify=model.get_labels())
    end = timer()
    print(end-start)

    #Trains models on test split
    start = timer()
    model.fit_grid_search( X_train, y_train )
    end = timer()
    print(end-start)


    #Print summary statistics from train
    model.report_model_results()

    all_models = [g.best_estimator_ for g in model.get_gs_results()]

    model.predict_and_score(all_models, X_test, y_test)

    '''
    [[ -7.72642022e-04  -9.33500647e-04   3.56949206e-04  -1.65679874e-04
   -1.90501419e-03   2.89065101e-01   0.00000000e+00]]
    Model Score: 0.977443609023 , Precision Score: 0.91847826087, Recall Score: 1.0
    '''
