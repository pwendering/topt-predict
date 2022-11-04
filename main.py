import ExtractSeqFeatures
import PredictKeyTemps
import matplotlib.pyplot as plt


def main(meltome_file):
    # print("Reading Meltome Atlas file")
    # pm = ParseMeltome(meltome_file)
    # pm.parse_meltome()

    ExtractSeqFeatures.extractSequenceFeatures("meltome_seqs_complete.fasta",
                                               features=["MISC", "AAC", "QSOrder", "CTDC", "PAAC"])

    x_data, y_data, x_ids, y_ids, x_vars, y_vars, datasets = PredictKeyTemps.prepare_data("aa_features.csv",
                                                                                          "curve_params.csv", 70)
    PredictKeyTemps.write_clean_data(x_data, y_data, datasets)
    x_data, y_data, datasets = PredictKeyTemps.read_clean_data()
    print("Number of samples: %d" % x_data.shape[0])
    print("Number of features: %d" % x_data.shape[1])

    X, y, datasets = PredictKeyTemps.remove_y_outliers_per_group(x_data, -y_data[:, 1], datasets)
    print("Reduced number of samples: %d" % X.shape[0])

    '''
    sns.boxplot(x=datasets, y=y)
    plt.xticks(rotation=90)
    plt.savefig("T_m_clean.png", bbox_inches='tight')'''

    '''PredictKeyTemps.svr_rbf(X, y)
    PredictKeyTemps.svr_linear(X, y)
    PredictKeyTemps.adaboost(X, y)
    PredictKeyTemps.xgboost_reg(X, y, datasets)
    PredictKeyTemps.mlpreg(X, y, datasets)
    PredictKeyTemps.bayesian_ridge(X, y, datasets)
    PredictKeyTemps.gbdt(X, y, datasets)
    PredictKeyTemps.cubist_reg(X, y, datasets)'''

    regr = PredictKeyTemps.randomforest(X, y, datasets, hyperparam=True, njobs=1)

    with open("aa_features.csv", "r") as f:
        features = f.readline().strip("\n").split(",")[1:]

    f_i = list(zip(features, regr.feature_importances_))
    f_i.sort(key=lambda x: x[1])
    plt.barh([x[0] for x in f_i[-31:-1]], [x[1] for x in f_i[-31:-1]])
    plt.savefig("rf_feature_importances_30.png", bbox_inches='tight')
    plt.close()

    # PredictKeyTemps.knn(X, y, datasets)


if __name__ == '__main__':
    main("cross-species.csv")


