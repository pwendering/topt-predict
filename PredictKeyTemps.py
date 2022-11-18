from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import pandas as pd
import re
from cubist import Cubist
import matplotlib.pyplot as plt
import seaborn as sb
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor


def read_x_data(x_data_file):
    print("Reading X data")
    x_df = pd.read_csv(x_data_file, index_col=0)
    x_matrix = x_df.to_numpy()
    x_ids = x_df.index.values
    x_vars = x_df.columns.values
    return x_matrix, x_ids, x_vars


def read_y_data(y_data_file):
    print("Reading Y data")
    y_df = pd.read_csv(y_data_file, index_col=1)
    datasets = y_df['dataset']
    y_df.drop(['dataset'], axis=1, inplace=True)
    y_matrix = y_df.to_numpy()
    y_ids = y_df.index.values
    y_vars = y_df.columns.values
    return y_matrix, y_ids, y_vars, datasets


def prepare_y_data(y_data, y_ids, x_ids, datasets):
    print("Preparing Y data")

    # find IDs that match with UniProt ID regex pattern
    y_ids = [find_uniprot_id(x) for x in y_ids]

    # re-arrange y-data according to ordering of IDs in x-data
    match_idx = [y_ids.index(x) for x in x_ids if x in y_ids]
    y_data = y_data[match_idx, :]
    y_ids = [y_ids[i] for i in match_idx]
    datasets = [datasets[i] for i in match_idx]
    return y_data, y_ids, datasets


def prepare_x_data(x_data, x_ids, y_ids):
    print("Preparing X data")
    match_idx = [np.where(x_ids == x)[0][0] for x in y_ids if x in x_ids]
    x_data = x_data[match_idx, :]
    x_ids = x_ids[match_idx]

    return x_data, x_ids


def prepare_data(x_data_file, y_data_file, pct_rmse=95):
    x_matrix, x_ids, x_vars = read_x_data(x_data_file)
    y_matrix, y_ids, y_vars, datasets = read_y_data(y_data_file)

    y_matrix, y_ids, datasets = prepare_y_data(y_matrix, y_ids, x_ids, datasets)
    x_matrix, x_ids = prepare_x_data(x_matrix, x_ids, y_ids)

    # remove rows with nan values and with insufficient RMSE (remove upper 5%, distribution is highly left-skewed)
    print("Cleaning datasets")
    nan_row_idx = [np.isnan(x_matrix[row, :]).any() or np.isnan(y_matrix[row, :]).any()
                   for row in range(0, x_matrix.shape[0])]
    rmse_col_idx = [x == "rmse" for x in y_vars]
    rmse_threshold = np.nanpercentile(y_matrix[:, rmse_col_idx], pct_rmse)
    low_rmse_idx = [x < rmse_threshold for x in y_matrix[:, rmse_col_idx]]

    keep_row_idx = [i for i in range(0, len(nan_row_idx)) if low_rmse_idx[i] and not nan_row_idx[i]]
    y_matrix = y_matrix[keep_row_idx, :]
    x_matrix = x_matrix[keep_row_idx, :]
    y_ids = [y_ids[i] for i in keep_row_idx]
    x_ids = [x_ids[i] for i in keep_row_idx]
    datasets = [datasets[i] for i in keep_row_idx]

    return x_matrix, y_matrix, x_ids, y_ids, x_vars, y_vars, datasets


def write_clean_data(x_data, y_data, datasets):
    print("Writing clean data to file")
    np.savetxt("X.csv", x_data, delimiter=",")
    np.savetxt("y.csv", y_data, delimiter=",")
    with open("groups.txt", "w", newline="\n") as f:
        for line in datasets:
            f.write(line + "\n")


def read_clean_data():
    print("Reading data from file")
    x_data = np.genfromtxt("X.csv", delimiter=",")
    y_data = np.genfromtxt("y.csv", delimiter=",")
    groups = []
    with open("groups.txt", "r") as f:
        for line in f:
            groups.append(line.strip("\n"))
    return x_data, y_data, groups


def remove_y_outliers_per_group(x_data, y_data, groups):
    print("Removing outliers for each group (outside median +/- 3*sd)")
    uniq_groups = set(groups)
    for g in uniq_groups:
        group_idx = np.asarray([i for i in range(0, len(groups)) if groups[i] == g])
        med, sd = np.median(y_data[group_idx]), np.std(y_data[group_idx])
        sd_threshold = 3 * sd
        t_lower, t_upper = med - sd_threshold, med + sd_threshold
        remove_idx = np.where(np.logical_or((t_lower > y_data[group_idx]), (y_data[group_idx] > t_upper)))[0]
        y_data = np.delete(y_data, group_idx[remove_idx])
        x_data = np.delete(x_data, group_idx[remove_idx], axis=0)
        for i in sorted(remove_idx, reverse=True):
            del groups[group_idx[i]]

    return x_data, y_data, groups


def split_data_train_test(x_data, y_data):
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(x_data, y_data, range(0, len(y_data)), test_size=0.4, random_state=42)
    return X_train, X_test, y_train, y_test, idx_test


def score_prediction(y_pred, y_test):
    print("Root mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))
    print("Mean absolute error: %.2f" % np.sqrt(mean_absolute_error(y_test, y_pred)))
    print("Mean absolute percentage error: %.2f" % np.sqrt(mean_absolute_percentage_error(y_test, y_pred)))
    print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))
    print("Pearson correlation: %.2f\n\n" % np.corrcoef(y_test, y_pred)[0][1])


def scatter_test_pred(y_pred, y_test, groups=None, fname="scatterplot.png"):
    df = pd.DataFrame({"y_test": y_test, "y_pred": y_pred, "groups": groups})
    ax = sb.scatterplot(data=df, x="y_test", y="y_pred", hue="groups")
    ax.set_ylim((0, 100))
    ax.set_xlim((0, 100))
    ax.set_xlabel("Test")
    ax.set_ylabel("Prediction")

    sb.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    f = plt.gcf()
    f.savefig(fname, bbox_inches='tight')
    plt.close(f)


def svr_rbf(x_data, y_data):
    print("Training SVR (RBF kernel)")
    regr = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)

    X_train, X_test, y_train, y_test, idx_test = split_data_train_test(x_data, y_data)

    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_test)
    score_prediction(y_pred, y_test)

    return regr


def svr_linear(x_data, y_data):
    print("Training SVR (linear kernel)")
    regr = SVR(kernel="linear", C=100, gamma="auto", epsilon=0.3)

    X_train, X_test, y_train, y_test, idx_test = split_data_train_test(x_data, y_data)

    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_test)
    score_prediction(y_pred, y_test)

    return regr


def svr_poly(x_data, y_data):
    print("Training SVR (polynomial kernel)")
    regr = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)

    X_train, X_test, y_train, y_test, idx_test = split_data_train_test(x_data, y_data)

    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_test)
    score_prediction(y_pred, y_test)

    return regr


def ols(x_data, y_data):
    print("OLS model")
    X_train, X_test, y_train, y_test, idx_test = split_data_train_test(x_data, y_data)
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    score_prediction(y_pred, y_test)


def ridge(x_data, y_data):
    print("Ridge regression model")
    X_train, X_test, y_train, y_test, idx_test = split_data_train_test(x_data, y_data)
    print("alpha = 0.5")
    regr = linear_model.Ridge(alpha=.5)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    score_prediction(y_pred, y_test)


def lasso(x_data, y_data):
    print("Lasso regression")
    X_train, X_test, y_train, y_test, idx_test = split_data_train_test(x_data, y_data)
    print("alpha = 0.01")
    regr = linear_model.Lasso(alpha=.01)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    score_prediction(y_pred, y_test)


def gbdt(x_data, y_data, groups=None):
    print("Gradient Boosted Decision Tree")
    X_train, X_test, y_train, y_test, idx_test = split_data_train_test(x_data, y_data)
    groups = [groups[i] for i in idx_test]
    regr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=42, loss='squared_error')
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    score_prediction(y_pred, y_test)
    scatter_test_pred(y_pred, y_test, groups, 'scatter_gbdt.png')


def adaboost(x_data, y_data):
    print("AdaBoost regression")
    X_train, X_test, y_train, y_test, idx_test = split_data_train_test(x_data, y_data)
    regr = AdaBoostRegressor(random_state=42, n_estimators=100)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    score_prediction(y_pred, y_test)


def randomforest(x_data, y_data, groups=None, hyperparam=False, njobs=1):
    print("Random Forest")
    X_train, X_test, y_train, y_test, idx_test = split_data_train_test(x_data, y_data)
    groups = [groups[i] for i in idx_test]
    regr = RandomForestRegressor(random_state=42, n_estimators=1400, min_samples_split=5, min_samples_leaf=2,
                                 max_features='sqrt', max_depth=None, bootstrap=False)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    score_prediction(y_pred, y_test)
    scatter_test_pred(y_pred, y_test, groups, 'scatter_rf.png')

    if hyperparam:
        print("Hyperparameter fitting")
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]  # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}

        rf_random = RandomizedSearchCV(estimator=regr, param_distributions=random_grid, n_iter=100, cv=10, verbose=2,
                                       random_state=42, n_jobs=njobs)
        rf_random.fit(X_train, y_train)
        print(rf_random.best_params_)
        y_pred = rf_random.best_estimator_.predict(X_test)
        score_prediction(y_pred, y_test)
        scatter_test_pred(y_pred, y_test, groups, 'scatter_rf_optimized.png')

    return regr


def knn(x_data, y_data, groups=None, k=10):
    print("K-Nearest Neighbor")
    X_train, X_test, y_train, y_test, idx_test = split_data_train_test(x_data, y_data)
    groups = [groups[i] for i in idx_test]
    neigh = KNeighborsRegressor(n_neighbors=k)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    score_prediction(y_pred, y_test)
    scatter_test_pred(y_pred, y_test, groups, 'scatter_knn.png')


def cubist_reg(x_data, y_data, groups=None):
    print("Cubist regression")
    X_train, X_test, y_train, y_test, idx_test = split_data_train_test(x_data, y_data)
    groups = [groups[i] for i in idx_test]
    regr = Cubist()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    score_prediction(y_pred, y_test)
    scatter_test_pred(y_pred, y_test, groups, 'scatter_cubist.png')
    return regr


def bayesian_ridge(x_data, y_data, groups=None):
    print("Bayesian ridge regression")
    X_train, X_test, y_train, y_test, idx_test = split_data_train_test(x_data, y_data)
    groups = [groups[i] for i in idx_test]
    regr = linear_model.BayesianRidge()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    score_prediction(y_pred, y_test)
    scatter_test_pred(y_pred, y_test, groups, 'scatter_bayesian_ridge.png')
    return regr


def xgboost_reg(x_data, y_data, groups=None):
    print("XGBoost regression")
    X_train, X_test, y_train, y_test, idx_test = split_data_train_test(x_data, y_data)
    groups = [groups[i] for i in idx_test]
    regr = XGBRegressor()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    score_prediction(y_pred, y_test)
    scatter_test_pred(y_pred, y_test, groups, 'scatter_xgboost.png')
    return regr

def mlpreg(x_data, y_data, groups=None):
    print("MLP regression")
    X_train, X_test, y_train, y_test, idx_test = split_data_train_test(x_data, y_data)
    groups = [groups[i] for i in idx_test]
    regr = MLPRegressor()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    score_prediction(y_pred, y_test)
    scatter_test_pred(y_pred, y_test, groups, 'scatter_xgboost.png')
    return regr


def find_uniprot_id(prot_id):
    match = re.search(r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}", str(prot_id))
    if match is not None:
        return match[0]
    else:
        return match


def assert_nan(a):
    for i in range(0, len(a)):
        if not np.isnan(a[i]):
            return True
    return False

