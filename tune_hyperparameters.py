import PredictKeyTemps


def main(n_jobs=4):
    """
    Find optimal hyperparameters for the Random Forest regression model with selected features.

    :param n_jobs: number of threads to use
    """
    # output file names
    x_data_clean_file = "training_data/X.csv"
    y_data_clean_file = "training_data/y.csv"
    groups_clean_file = "training_data/groups.txt"
    x_scaler_file = "model/x_scaler.pkl"
    feature_ranking_rfecv_file = "feature_selection/feature_ranking_support.csv"
    rfecv_result_file = "training_data/rfecv_results.csv"

    # read clean data from file
    x_data, y_data, datasets = PredictKeyTemps.read_clean_data(x_data_clean_file, y_data_clean_file, groups_clean_file)

    # remove outliers (outside of three standard deviations from the median)
    X, y, datasets = PredictKeyTemps.remove_y_outliers_per_group(x_data, -y_data.loc[:, 'beta_param_2'], datasets)

    # Random Forest regression
    PredictKeyTemps.randomforest(X, y, datasets, optimal=False, hyperparam=True,
                                        fselect=False, n_jobs=n_jobs, rfecv_result_file=rfecv_result_file,
                                        feature_support_file=feature_ranking_rfecv_file, x_scaler_file=x_scaler_file)


if __name__ == '__main__':
    main(n_jobs=4)