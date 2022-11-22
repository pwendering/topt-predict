import ExtractSeqFeatures
import PredictKeyTemps
from ParseMeltome import ParseMeltome
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
import pandas as pd


def main(meltome_file, pct_rmse=70, parse_meltome=False, extract_features=False, train_hyperparams=False,
         print_feature_importance=False, select_features=False, n_jobs=1):
    if parse_meltome:
        print("Reading Meltome Atlas file")
        pm = ParseMeltome(meltome_file)
        pm.parse_meltome()

    if extract_features:
        # Extract features from amino acid sequences
        ExtractSeqFeatures.extractSequenceFeatures("meltome_seqs_complete.fasta",
                                                   features=["MISC", "AAC", "QSOrder", "CTDC", "PAAC"])

        #  clean up datasets
        x_data, y_data, x_ids, y_ids, x_vars, y_vars, datasets = PredictKeyTemps.prepare_data("aa_features.csv",
                                                                                              "curve_params.csv",
                                                                                              pct_rmse=pct_rmse)

        # write clean data to file
        PredictKeyTemps.write_clean_data(x_data, y_data, datasets)

    # read clean data from file
    x_data, y_data, datasets = PredictKeyTemps.read_clean_data()

    print("Number of samples: %d" % x_data.shape[0])
    print("Number of features: %d" % x_data.shape[1])

    # remove outliers (outside of three standard deviations from the median)
    X, y, datasets = PredictKeyTemps.remove_y_outliers_per_group(x_data, -y_data[:, 1], datasets)
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

    '''regr = PredictKeyTemps.randomforest(X, y, datasets, hyperparam=train_hyperparams, njobs=n_jobs,
                                        optimal=True)'''

    regr = PredictKeyTemps.randomforest(X, y, datasets, optimal=True, hyperparam=train_hyperparams,
                                        fselect=select_features, n_jobs=n_jobs)


    '''# cross validation for predictor performance
    # Train random forest regressor
    
    # cross-validation
    scoring = ['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error',
               'r2']
    scaler = StandardScaler().fit(X)
    X_transformed = scaler.transform(X)
    scores = cross_validate(regr, X_transformed, y, scoring=scoring, cv=10, n_jobs=n_jobs)
    pd.DataFrame(scores).to_csv("scores_rf.csv")'''

    if print_feature_importance:
        with open("aa_features.csv", "r") as f:
            features = f.readline().strip("\n").split(",")[1:]

        f_i = list(zip(features, regr.feature_importances_))
        f_i.sort(key=lambda x: x[1])
        plt.barh([x[0] for x in f_i[-31:-1]], [x[1] for x in f_i[-31:-1]])
        plt.savefig("rf_feature_importances_30.png", bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    main(
        "cross-species.csv",
        pct_rmse=70,
        parse_meltome=False,
        extract_features=False,
        train_hyperparams=False,
        select_features=True,
        print_feature_importance=False,
        n_jobs=1
    )


