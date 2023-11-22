import PredictKeyTemps
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import KFold, cross_validate


def main(n_jobs=4):
    """
    Create Random Forest model with optimal hyperparameters and save it.
    Feature importances will be plotted.
    Five-fold cross-valiation will be performed to evaluate the performance.

    :param n_jobs: number of threads to use
    :return:
    """
    # output file names
    x_data_clean_file = "training_data/X.csv"
    y_data_clean_file = "training_data/y.csv"
    groups_clean_file = "training_data/groups.txt"
    model_output_file = "model/random_forest_opt_model.sav"
    x_scaler_file = "model/x_scaler.pkl"
    feature_importance_barplot_file = "feature_selection/rf_feature_importances_30.png"
    feature_importance_file = "feature_selection/feature_importances_rf.tsv"
    feature_ranking_rfecv_file = "feature_selection/feature_ranking_support.csv"
    rfecv_result_file = "training_data/rfecv_results.csv"
    rf_score_file = "scores_rf_optimal.csv"

    # read clean data from file
    x_data, y_data, datasets = PredictKeyTemps.read_clean_data(x_data_clean_file, y_data_clean_file, groups_clean_file)

    # remove outliers (outside of three standard deviations from the median)
    X, y, datasets = PredictKeyTemps.remove_y_outliers_per_group(x_data, -y_data.loc[:, 'beta_param_2'], datasets)

    # Random Forest regression
    regr = PredictKeyTemps.randomforest(X, y, datasets, optimal=True, hyperparam=False,
                                        fselect=False, n_jobs=n_jobs, rfecv_result_file=rfecv_result_file,
                                        feature_support_file=feature_ranking_rfecv_file, x_scaler_file=x_scaler_file)

    # save regression model
    pickle.dump(regr, open(model_output_file, 'wb'))

    # Feature importances
    f_i = list(zip(regr.feature_names_in_, regr.feature_importances_))
    std = np.std([tree.feature_importances_ for tree in regr.estimators_], axis=0)
    sort_idx = np.argsort([-x[1] for x in f_i])
    f_i = [f_i[i] for i in sort_idx]
    std = [std[i] for i in sort_idx]
    plt.barh([x[0] for x in f_i[0:29]], [x[1] for x in f_i[0:29]], xerr=std[0:29])
    plt.xlabel("Mean decrease in impurity")
    plt.gca().invert_yaxis()
    plt.savefig(feature_importance_barplot_file, bbox_inches='tight')
    plt.close()

    with open(feature_importance_file, "w") as f:
        for fi in f_i:
            f.write(fi[0] + "\t" + str(fi[1]) + "\n")

    # cross validation for performance evaluation
    scoring = ['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error',
               'r2']
    scaler = pickle.load(open(x_scaler_file, 'rb'))
    X_transformed = scaler.transform(X)
    kfsplit = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(regr, X_transformed, y, scoring=scoring, cv=kfsplit, n_jobs=n_jobs, pre_dispatch=n_jobs,
                            verbose=2)
    pd.DataFrame(scores).to_csv(rf_score_file)


if __name__ == '__main__':
    main(n_jobs=4)
