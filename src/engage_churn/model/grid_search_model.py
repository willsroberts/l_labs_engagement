from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def grid_search_dict():
    '''
    defines all the models and hyperparameters to optimize with grid search.
    returns a list of tuples - (model, dict-of-hyper_params)
    '''
    #
    # gd_boost = {
    #     'learning_rate':[1, 0.05],
    #     'max_depth':[2,4],
    #     'max_features':['sqrt', 'log2'],
    #     'n_estimators':[50, 100]}
    #
    # ada_boost = {
    #     'learning_rate':[1, 0.05],
    #     'base_estimator__max_depth':[2,4],
    #     'base_estimator__max_features':['sqrt', 'log2'],
    #     'n_estimators':[50, 100]}

    random_forest_grid = {
        'n_estimators': [50],
        'max_features': ['sqrt', 'log2'],
        'min_samples_leaf': [1, 2],
        }

    return [
        # (GradientBoostingClassifier(), gd_boost),
        # (AdaBoostClassifier(DecisionTreeClassifier()), ada_boost),
        (RandomForestClassifier(), random_forest_grid)
    ]
