import ExtractSeqFeatures
import PredictKeyTemps
from ParseMeltome import ParseMeltome
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def main(meltome_file, pct_rmse=70, parse_meltome=False, extract_features=False, train_hyperparams=False,
         select_features=False, n_jobs=1):

    if parse_meltome:
        print("Reading Meltome Atlas file")
        pm = ParseMeltome(meltome_file)
        pm.parse_meltome()

    if extract_features:
        # Extract features from amino acid sequences
        '''ExtractSeqFeatures.extractSequenceFeatures("meltome_seqs_complete.fasta",
                                                   features=["MISC", "AAC", "QSOrder", "CTDC", "PAAC", "AC", "CTriad", "DistancePair", "GAAC", "Geary",
                                                       "Moran", "SOCNumber", "CKSAAP type 1", "CTDD", "DPC type 1", "GDPC type 1", "NMBroto",
                                                       "PseKRAAC type 10"])'''

        #  clean up datasets
        x_data, y_data, x_ids, y_ids, x_vars, y_vars, datasets = PredictKeyTemps.prepare_data("aa_features_selected.csv",
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

    regr = PredictKeyTemps.randomforest(X, y, datasets, optimal=False, hyperparam=train_hyperparams,
                                        fselect=select_features, n_jobs=n_jobs)

    if select_features:
        PredictKeyTemps.write_rfecv_features("feature_ranking_support.csv", "feature_names.txt")
        PredictKeyTemps.write_selected_features("feature_ranking_support.csv", "aa_features.csv", "aa_features_selected.csv")
        PredictKeyTemps.plot_rfecv_scores("rfecv_results.csv", 2839, 10)


    # cross validation for predictor performance
    # Train random forest regressor
    '''
    regr = RandomForestRegressor(random_state=42, n_jobs=n_jobs)
    scoring = ['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error',
               'r2']
    scaler = StandardScaler().fit(X)
    X_transformed = scaler.transform(X)
    kfsplit = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(regr, X_transformed, y, scoring=scoring, cv=kfsplit, n_jobs=n_jobs, pre_dispatch=n_jobs, verbose=2)
    pd.DataFrame(scores).to_csv("scores_rf.csv")
    '''

if __name__ == '__main__':
    main(
        "cross-species.csv",
        pct_rmse=70,
        parse_meltome=False,
        extract_features=False,
        train_hyperparams=True,
        select_features=False,
        n_jobs=5
    )


