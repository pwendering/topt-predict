import ExtractSeqFeatures
import PredictKeyTemps


def main(t_r2=0.6):
    """
    Extract features for all sequences associated with the measured proteins in the Meltome Atlas (Jarząb et al. 2020).

    :param t_r2: threshold for R2 score to apply to tpp fits

    """

    # input file names
    meltome_seq_file = "training_data/meltome_seqs.fasta"
    tpp_fits_file = "training_data/curve_params.csv"

    # output file names
    # aa_feature_selected_file = "training_data/aa_features.csv"
    aa_feature_selected_file = "training_data/aa_features_selected.csv"
    x_data_clean_file = "training_data/X.csv"
    y_data_clean_file = "training_data/y.csv"
    groups_clean_file = "training_data/groups.txt"

    # features to extract
    features = ["MISC", "AAC", "QSOrder", "CTDC", "PAAC", "AC", "CTriad", "DistancePair", "GAAC", "Geary", "Moran",
                "SOCNumber", "CKSAAP type 1", "CTDD", "DPC type 1", "GDPC type 1", "NMBroto", "PseKRAAC type 10"]

    # Extract features from amino acid sequences
    ExtractSeqFeatures.extractSequenceFeatures(meltome_seq_file, features=features)

    #  clean up datasets
    x_data, y_data, datasets = PredictKeyTemps.prepare_data(aa_feature_selected_file, tpp_fits_file, t_r2=t_r2)

    # write clean data to file
    PredictKeyTemps.write_clean_data(x_data, y_data, datasets, x_data_clean_file, y_data_clean_file,
                                     groups_clean_file)


if __name__ == '__main__':
    main(t_r2=0.6)
