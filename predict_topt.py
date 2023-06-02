import argparse
import ExtractSeqFeatures
import PredictKeyTemps
import os
import pandas as pd
import pickle


def main(seq_file, model_file, extract_features_flag, predict_flag):

    features = ["MISC", "AAC", "QSOrder", "CTDC", "PAAC", "AC", "CTriad", "DistancePair", "GAAC", "Geary", "Moran",
                "SOCNumber", "CKSAAP type 1", "CTDD", "DPC type 1", "GDPC type 1", "NMBroto", "PseKRAAC type 10"]

    if extract_features_flag:
        inputdatadir = os.path.dirname(seq_file)
        ExtractSeqFeatures.extractSequenceFeatures(seq_file, features=features, hist_plot_flag=False)
        PredictKeyTemps.write_selected_features("feature_selection/feature_ranking_support.csv",
                                                inputdatadir + "/aa_features.csv",
                                                inputdatadir + "/aa_features_selected.csv")

    if predict_flag:
        inputdatadir = os.path.dirname(seq_file)
        x = pd.read_csv(inputdatadir + "/aa_features_selected.csv", index_col=0)

        # standardize data
        scaler = pickle.load(open('model/x_scaler.pkl', 'rb'))
        x_scaled = scaler.transform(x)

        # load regression model
        regr = pickle.load(open(model_file, 'rb'))

        topt = regr.predict(x_scaled)

        df = pd.DataFrame(topt, index=x.index, columns=["Topt"])

        df.to_csv(inputdatadir + "/topt_predicted.csv", lineterminator="\n")


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(
        prog='predict_topt',
        description='Predition of optimal protein temperatures based on the amino acid sequence')

    parser.add_argument('seq_file', type=str, help='fasta-formatted file containing amino acid sequences')
    parser.add_argument('-m', '--model', type=str, help='regression model file (.sav)',
                        default="model/random_forest_opt_model.sav")
    parser.add_argument('-e', '--extract_features', help='extract features from amino acid sequences (will be stored ' +
                                                         'in same directory as sequence file)',
                        action='store_true', default=False)
    parser.add_argument('-p', '--predict', help='predict key temperatures', action='store_true', default=True)

    args = parser.parse_args()

    main(args.seq_file, args.model, args.extract_features, args.predict)
