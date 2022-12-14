import ExtractSeqFeatures
import PredictKeyTemps
import matplotlib.pyplot as plt
import numpy as np
import pickle


def main(pct_rmse=70, extract_features=False, train_hyperparams=False,
         select_features=False, n_jobs=1):

    if extract_features:
        features = ["MISC", "AAC", "QSOrder", "CTDC", "PAAC", "AC", "CTriad", "DistancePair", "GAAC", "Geary", "Moran",
                    "SOCNumber", "CKSAAP type 1", "CTDD", "DPC type 1", "GDPC type 1", "NMBroto", "PseKRAAC type 10"]

        # Extract features from amino acid sequences
        ExtractSeqFeatures.extractSequenceFeatures("training_data/meltome_seqs.fasta", features=features)

        #  clean up datasets
        x_data, y_data, datasets = PredictKeyTemps.prepare_data("training_data/aa_features_selected.csv",
                                                                "training_data/curve_params.csv",
                                                                pct_rmse=pct_rmse)

        # write clean data to file
        PredictKeyTemps.write_clean_data(x_data, y_data, datasets, "training_data/X.csv", "training_data/y.csv",
                                         "training_data/groups.txt")

    # read clean data from file
    x_data, y_data, datasets = PredictKeyTemps.read_clean_data("training_data/X.csv", "training_data/y.csv",
                                         "training_data/groups.txt")

    print("Number of samples: %d" % x_data.shape[0])
    print("Number of features: %d" % x_data.shape[1])

    # remove outliers (outside of three standard deviations from the median)
    X, y, datasets = PredictKeyTemps.remove_y_outliers_per_group(x_data, -y_data.loc[:, 'beta_param_2'], datasets)
    print("Reduced number of samples: %d" % X.shape[0])

    print("\n\nTesting different regression approaches:\n")
    print("Approach\tRMSE\tMAE\tMAPE\tR2\trhoP")
    '''PredictKeyTemps.lasso(X, y)
    PredictKeyTemps.ridge(X, y)
    PredictKeyTemps.svr_rbf(X, y)
    PredictKeyTemps.svr_linear(X, y)
    PredictKeyTemps.adaboost(X, y)
    PredictKeyTemps.xgboost_reg(X, y, datasets)
    PredictKeyTemps.mlpreg(X, y, datasets)
    PredictKeyTemps.bayesian_ridge(X, y, datasets)
    PredictKeyTemps.gbdt(X, y, datasets)
    PredictKeyTemps.cubist_reg(X, y, datasets)
    PredictKeyTemps.knn(X, y, datasets)'''

    regr = PredictKeyTemps.randomforest(X, y, datasets, optimal=True, hyperparam=train_hyperparams,
                                        fselect=select_features, n_jobs=n_jobs)

    # save regression model
    pickle.dump(regr, open("model/random_forest_opt_model_top.sav", 'wb'))

    # Feature importances
    f_i = list(zip(regr.feature_names_in_, regr.feature_importances_))
    std = np.std([tree.feature_importances_ for tree in regr.estimators_], axis=0)
    sort_idx = np.argsort([-x[1] for x in f_i])
    f_i = [f_i[i] for i in sort_idx]
    std = [std[i] for i in sort_idx]
    plt.barh([x[0] for x in f_i[0:29]], [x[1] for x in f_i[0:29]], xerr=std[0:29])
    plt.xlabel("Mean decrease in impurity")
    plt.gca().invert_yaxis()
    plt.savefig("feature_selection/rf_feature_importances_30.png", bbox_inches='tight')
    plt.close()

    with open("feature_selection/feature_importances_rf.tsv", "w") as f:
        for fi in f_i:
            f.write(fi[0] + "\t" + str(fi[1]) + "\n")


    if select_features:
        PredictKeyTemps.write_rfecv_features("feature_selection/feature_ranking_support.csv",
                                             "feature_selection/feature_names.txt")
        PredictKeyTemps.write_selected_features("feature_selection/feature_ranking_support.csv",
                                                "training_data/aa_features.csv",
                                                "training_data/aa_features_selected.csv")
        PredictKeyTemps.plot_rfecv_scores("training_data/rfecv_results.csv", 2839, 10)


    # cross validation for predictor performance
    '''scoring = ['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error',
               'r2']
    scaler = StandardScaler().fit(X)
    X_transformed = scaler.transform(X)
    kfsplit = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(regr, X_transformed, y, scoring=scoring, cv=kfsplit, n_jobs=n_jobs, pre_dispatch=n_jobs, verbose=2)
    pd.DataFrame(scores).to_csv("scores_rf_optimal.csv")'''


if __name__ == '__main__':
    main(
        pct_rmse=70,
        extract_features=False,
        train_hyperparams=False,
        select_features=False,
        n_jobs=4
    )