import PredictKeyTemps


def main():
    """
    Compare different regression models with default parameters, using the reduced set of features.

    """
    # output file names
    x_data_clean_file = "training_data/X.csv"
    y_data_clean_file = "training_data/y.csv"
    groups_clean_file = "training_data/groups.txt"
    x_scaler_file = "model/x_scaler.pkl"

    # read clean data from file
    x_data, y_data, datasets = PredictKeyTemps.read_clean_data(x_data_clean_file, y_data_clean_file, groups_clean_file)

    # remove outliers (outside of three standard deviations from the median)
    X, y, datasets = PredictKeyTemps.remove_y_outliers_per_group(x_data, -y_data.loc[:, 'beta_param_2'], datasets)

    # Compare different regression models with default settings
    print("\n\nTesting different regression approaches:\n")
    print("Approach\tRMSE\tMAE\tMAPE\tR2\trhoP")

    PredictKeyTemps.lasso(X, y)
    PredictKeyTemps.ridge(X, y)
    PredictKeyTemps.svr_rbf(X, y)
    PredictKeyTemps.svr_linear(X, y)
    PredictKeyTemps.adaboost(X, y)
    PredictKeyTemps.xgboost_reg(X, y, datasets)
    PredictKeyTemps.mlpreg(X, y, datasets)
    PredictKeyTemps.bayesian_ridge(X, y, datasets)
    PredictKeyTemps.gbdt(X, y, datasets)
    PredictKeyTemps.cubist_reg(X, y, datasets)
    PredictKeyTemps.knn(X, y, datasets)
    PredictKeyTemps.randomforest(X, y, datasets, x_scaler_file=x_scaler_file, optimal=False)


if __name__ == '__main__':
    main()
