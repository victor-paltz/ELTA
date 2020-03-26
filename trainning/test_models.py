from sklearn import ensemble, linear_model, metrics, naive_bayes
from sklearn.metrics import (accuracy_score, auc, confusion_matrix,
                             make_scorer, precision_recall_curve,
                             precision_score, recall_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)


def finetune_MultinomialNB(train_x, train_y, valid_x, valid_y):

    print("Grid search on multinomial naive bayes classifier")

    clf = naive_bayes.MultinomialNB()

    param_grid = {"alpha": [0, 0.05, .1, .2, 1.0]}

    skf = StratifiedKFold(n_splits=5)

    grid_search = GridSearchCV(clf, param_grid, scoring="f1_weighted",
                               cv=skf, return_train_score=True)

    grid_search.fit(train_x, train_y)

    # compute scores on validation
    y_pred = grid_search.predict(valid_x)
    weighted_f1 = metrics.f1_score(valid_y, y_pred, average='weighted')
    print(" -> Weighted F1 score:", weighted_f1)

    print(' -> Best params:')
    print(grid_search.best_params_)

    return grid_search.best_params_


def finetune_LogisticRegression(train_x, train_y, valid_x, valid_y):

    print("Grid search on LogisticRegression classifier")

    clf = linear_model.LogisticRegression()

    param_grid = {
        "penalty": ["l2"],
        "C": [1, 1.5, 2, 4]
    }

    skf = StratifiedKFold(n_splits=5)

    grid_search = GridSearchCV(clf, param_grid, scoring="f1_weighted",
                               cv=skf, return_train_score=True)

    grid_search.fit(train_x, train_y)

    # compute scores on validation
    y_pred = grid_search.predict(valid_x)
    weighted_f1 = metrics.f1_score(valid_y, y_pred, average='weighted')
    print(" -> Weighted F1 score:", weighted_f1)

    print(' -> Best params:')
    print(grid_search.best_params_)

    return grid_search.best_params_


def finetune_AdaBoost(train_x, train_y, valid_x, valid_y):

    print("Grid search on AdaBoost classifier")

    clf = ensemble.AdaBoostClassifier(n_estimators=100)

    param_grid = {
        "learning_rate": [.1, 1, 5],
        "n_estimators": [20, 50, 100, 300]
    }

    skf = StratifiedKFold(n_splits=5)

    grid_search = GridSearchCV(clf, param_grid, scoring="f1_weighted",
                               cv=skf, return_train_score=True)

    grid_search.fit(train_x, train_y)

    # compute scores on validation
    y_pred = grid_search.predict(valid_x)
    weighted_f1 = metrics.f1_score(valid_y, y_pred, average='weighted')
    print(" -> Weighted F1 score:", weighted_f1)

    print(' -> Best params:')
    print(grid_search.best_params_)

    return grid_search.best_params_


def finetune_RandomForestClassifier(train_x, train_y, valid_x, valid_y):

    print("Grid search on RandomForest classifier")

    clf = ensemble.RandomForestClassifier()

    param_grid = {
        'min_samples_split': [3, 5],
        'n_estimators': [100, 300],
        'max_depth': [3, 5, 15],
        'max_features': [3, 5]
    }

    skf = StratifiedKFold(n_splits=5)
    grid_search = GridSearchCV(clf, param_grid, scoring="f1_weighted",
                               cv=skf, return_train_score=True)

    grid_search.fit(train_x, train_y)

    # make the predictions
    y_pred = grid_search.predict(valid_x)
    weighted_f1 = metrics.f1_score(valid_y, y_pred, average='weighted')
    print(" -> Weighted F1 score:", weighted_f1)

    print('Best params:')
    print(grid_search.best_params_)
