import PredictKeyTemps

def main(n_jobs=4):
    """
    Select features using Random Forest regression.
    The input should be the full set of extracted features (i.e., X.csv generated from "aa_features.csv").

    :param n_jobs: number of threads to use

    """

    # input file names
    x_data_clean_file = "training_data/X.csv"
    y_data_clean_file = "training_data/y.csv"
    groups_clean_file = "training_data/groups.txt"

    # output file names
    aa_feature_file = "training_data/aa_features.csv"
    aa_feature_selected_file = "training_data/aa_features_selected.csv"
    x_scaler_file = "model/x_scaler.pkl"
    feature_ranking_rfecv_file = "feature_selection/feature_ranking_support.csv"
    feature_names_file = "feature_selection/feature_names.txt"
    rfecv_result_file = "training_data/rfecv_results.csv"

    # read clean data from file
    x_data, y_data, datasets = PredictKeyTemps.read_clean_data(x_data_clean_file, y_data_clean_file, groups_clean_file)

    # remove outliers (outside of three standard deviations from the median)
    X, y, datasets = PredictKeyTemps.remove_y_outliers_per_group(x_data, -y_data.loc[:, 'beta_param_2'], datasets)

    # Random Forest regression
    PredictKeyTemps.randomforest(X, y, datasets, optimal=False, hyperparam=False, fselect=True, n_jobs=n_jobs,
                                 rfecv_result_file=rfecv_result_file, feature_support_file=feature_ranking_rfecv_file,
                                 x_scaler_file=x_scaler_file)

    # write selected features to file
    PredictKeyTemps.write_rfecv_features(feature_ranking_rfecv_file,
                                         feature_names_file)
    PredictKeyTemps.write_selected_features(feature_ranking_rfecv_file,
                                            aa_feature_file,
                                            aa_feature_selected_file)
    PredictKeyTemps.plot_rfecv_scores(rfecv_result_file, X.shape[1], 10)


if __name__ == '__main__':
    main(n_jobs=4)
