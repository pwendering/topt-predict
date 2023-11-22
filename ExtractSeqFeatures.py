import iFeatureOmegaCLI
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import os.path as op
from protlearn.features import entropy
import os

def extractSequenceFeatures(fasta_file, features=None, hist_plot_flag=True):
    print("Extracting sequence features")

    outfilepath = os.path.dirname(fasta_file)

    if features is None:
        features = ["AAC", "MISC"]

    # set up iProtein instance if at least one feature is requested
    if not features == "MISC":
        protein = iFeatureOmegaCLI.iProtein(fasta_file)

    # disable plotting if MISC is not contained in features
    if hist_plot_flag and ("MISC" not in features):
        hist_plot_flag = False

    # initialize dataframe for protein features extracted with iFeature
    df_prot = pd.DataFrame()

    for feat in features:

        if feat == "MISC":
            df_misc, ids, seq_idx = get_misc_features(fasta_file)
        elif feat == "protdcal_prot" and check_file_exists(os.path.join(outfilepath, feat + ".csv")):
            df = pd.read_csv(feat + ".csv")
            # remove first column (protein IDs)
            df.drop(columns=df.columns[0], axis=1, inplace=True)
            df_prot = pd.concat([df_prot, df], axis=1)
        else:
            if not check_valid_feature(feat):
                print("%s is not a valid feature" % feat)
                continue

            print("=> Processing feature: %s" % feat)
            fname = os.path.join(outfilepath, feat.replace(" ", "_") + ".csv")
            df_csv_bool = True
            if not check_file_exists(fname):
                # perform calculation
                protein.get_descriptor(feat)
                # write results to CSV file
                df_csv_bool = protein.to_csv(fname, "index=False", header=True)

            if df_csv_bool:
                # read from CSV file
                df = pd.read_csv(fname)
                # remove first column (protein IDs)
                df.drop(columns=df.columns[0], axis=1, inplace=True)
                df_prot = pd.concat([df_prot, df], axis=1)
            else:
                print("Feature %s could not be extracted!" % feat)


    # combine dataframes
    if features == "MISC":
        df_misc['ID'] = ids
        df_full = df_misc
    elif "MISC" in features:
        cols = list(df_misc.columns)
        df_misc['ID'] = ids
        df_misc = df_misc[['ID'] + cols]
        df_full = pd.concat([df_misc, df_prot.iloc[seq_idx, :]], axis=1)
    else:
        df_full = df_prot

    # write features
    df_full.to_csv(os.path.join(outfilepath, "aa_features.csv"), sep=",", lineterminator="\n", index=False)

    # write only feature names
    with open(os.path.join(outfilepath, "feature_names.txt"), "w", newline="\n") as f:
        for i in range(1, len(df_full.columns)):
            f.write(df_full.columns[i] + "\n")

    if hist_plot_flag:
        plot_histograms_misc(df_misc)


def get_misc_features(fasta_file, seq_limit=50000):
    # sequence length, molecular weight, isoelectric point, aromaticity, instability index,
    # secondary structure fraction (helix, turn, sheet)
    outfilepath = os.path.dirname(fasta_file)
    fname = os.path.join(outfilepath, "misc.csv")
    if not check_file_exists(fname):
        feat_lists = []
        ids = []
        seq_idx = []
        seq_counter = 0
        with open(fasta_file, "r") as f:
            # read first ID
            ID = f.readline().strip(">").strip("\n")
            while ID != "":
                # read current sequence
                seq = f.readline().strip("\n")
                seq_counter += 1
                seq_length = len(seq)
                # remove X from sequence as ProteinAnalysis cannot handle X
                prot = ProteinAnalysis(seq.replace("X", "").replace("U", ""))
                sec_str_fract = prot.secondary_structure_fraction()
                group_freqs = count_group_fequencies(seq)
                if seq_length <= seq_limit:  # 2285 = 99% percentile
                    seq_idx.append(seq_counter - 1)
                    ids.append(ID)
                    entries = [seq_length,
                               prot.molecular_weight(),
                               prot.isoelectric_point(),
                               prot.aromaticity(),
                               prot.instability_index(),
                               sec_str_fract[0],
                               sec_str_fract[1],
                               sec_str_fract[2],
                               1-np.sum(sec_str_fract),
                               prot.molar_extinction_coefficient()[0],
                               prot.gravy(),
                               entropy(seq.replace("X", "").replace("U", "")),
                               group_freqs[0],
                               group_freqs[1],
                               group_freqs[2],
                               group_freqs[3],
                               group_freqs[4]]
                    feat_lists.append(entries)
                # read next ID
                ID = f.readline().strip(">").strip("\n")
        # create dataframe of features
        cols = ["SEQL", "MW", "ISOEP", "AROM", "INST", "HELIXF", "TURNF", "SHEETF", "REMAINF", "MEXTC_1", "GRAVY",
                "ENTROPY", "G1", "G2", "G3", "G4", "G5"]
        feat_array = np.asarray(feat_lists)
        del feat_lists
        # log10-transform SEQL, MW, and MEXTC_1
        log_col_idx = [x in ["SEQL", "MW", "MEXTC_1"] for x in cols]
        feat_array[:, log_col_idx] = np.log10(feat_array[:, log_col_idx])
        # remove inf values (some MEXTC_1 values are zero)
        feat_array[np.where(np.isinf(feat_array))] = np.nan
        df_misc = pd.DataFrame(feat_array, columns=cols)
        df_write = df_misc.copy()
        df_write['ID'] = ids
        df_write['idx'] = seq_idx
        df_write.to_csv(fname, lineterminator="\n")
    else:
        df_misc = pd.read_csv(fname)
        ids = df_misc['ID']
        seq_idx = df_misc['idx']
        df_misc.drop(columns=df_misc.columns[0], axis=1, inplace=True)
        df_misc.drop(['ID'], axis=1, inplace=True)
        df_misc.drop(['idx'], axis=1, inplace=True)

    return df_misc, ids, seq_idx


def plot_histograms_misc(df):
    df.drop(['ID'], axis=1, inplace=True)
    cols = df.columns
    print("Plotting histograms")
    for i in range(0, len(cols)):
        print("\t" + cols[i])
        sb.histplot(df[cols[i]])
        f = plt.gcf()
        f.savefig("histograms/" + cols[i] + "_histogram.png", bbox_inches='tight')
        plt.clf()


def count_group_fequencies(seq):
    groups = {
        "hydrophobic": ["V", "I", "L", "F", "M", "W", "Y", "C"],
        "negatively charged": ["D", "E"],
        "positively charged": ["R", "K", "H"],
        "conformational": ["G", "P"],
        "polar": ["N", "Q", "S"],
        "others": ["A", "T"]
    }
    seq_length = len(seq)
    group_freqs = []
    for g in groups.keys():#
        group_freqs.append(np.sum([seq.count(groups[g][i]) for i in range(0, len(groups[g]))])/seq_length)

    return np.asarray(group_freqs)


def check_valid_feature(feature):
    cmd_dict = get_cmd_dict()
    if feature in cmd_dict.keys():
        return True
    else:
        return False


def check_file_exists(fname):
    return op.exists(fname)


def get_cmd_dict():
    return {
        'AAC': 'self._AAC()',
        'EAAC': 'self._EAAC()',
        'CKSAAP type 1': 'self._CKSAAP(normalized=True)',
        'CKSAAP type 2': 'self._CKSAAP(normalized=False)',
        'DPC type 1': 'self._DPC(normalized=True)',
        'DPC type 2': 'self._DPC(normalized=False)',
        'DDE': 'self._DDE()',
        'TPC type 1': 'self._TPC(normalized=True)',
        'TPC type 2': 'self._TPC(normalized=False)',
        'binary': 'self._binary()',
        'binary_6bit': 'self._binary_6bit()',
        'binary_5bit type 1': 'self._binary_5bit_type_1()',
        'binary_5bit type 2': 'self._binary_5bit_type_2()',
        'binary_3bit type 1': 'self._binary_3bit_type_1()',
        'binary_3bit type 2': 'self._binary_3bit_type_2()',
        'binary_3bit type 3': 'self._binary_3bit_type_3()',
        'binary_3bit type 4': 'self._binary_3bit_type_4()',
        'binary_3bit type 5': 'self._binary_3bit_type_5()',
        'binary_3bit type 6': 'self._binary_3bit_type_6()',
        'binary_3bit type 7': 'self._binary_3bit_type_7()',
        'AESNN3': 'self._AESNN3()',
        'GAAC': 'self._GAAC()',
        'EGAAC': 'self._EGAAC()',
        'CKSAAGP type 1': 'self._CKSAAGP(normalized=True)',
        'CKSAAGP type 2': 'self._CKSAAGP(normalized=False)',
        'GDPC type 1': 'self._GDPC(normalized=True)',
        'GDPC type 2': 'self._GDPC(normalized=False)',
        'GTPC type 1': 'self._GTPC(normalized=True)',
        'GTPC type 2': 'self._GTPC(normalized=False)',
        'AAIndex': 'self._AAIndex()',
        'ZScale': 'self._ZScale()',
        'BLOSUM62': 'self._BLOSUM62()',
        'NMBroto': 'self._NMBroto()',
        'Moran': 'self._Moran()',
        'Geary': 'self._Geary()',
        'CTDC': 'self._CTDC()',
        'CTDT': 'self._CTDT()',
        'CTDD': 'self._CTDD()',
        'CTriad': 'self._CTriad()',
        'KSCTriad': 'self._KSCTriad()',
        'SOCNumber': 'self._SOCNumber()',
        'QSOrder': 'self._QSOrder()',
        'PAAC': 'self._PAAC()',
        'APAAC': 'self._APAAC()',
        'OPF_10bit': 'self._OPF_10bit()',
        'OPF_10bit type 1': 'self._OPF_10bit_type_1()',
        'OPF_7bit type 1': 'self._OPF_7bit_type_1()',
        'OPF_7bit type 2': 'self._OPF_7bit_type_2()',
        'OPF_7bit type 3': 'self._OPF_7bit_type_3()',
        'ASDC': 'self._ASDC()',
        'DistancePair': 'self._DistancePair()',
        'AC': 'self._AC()',
        'CC': 'self._CC()',
        'ACC': 'self._ACC()',
        'PseKRAAC type 1': 'self._PseKRAAC_type_1()',
        'PseKRAAC type 2': 'self._PseKRAAC_type_2()',
        'PseKRAAC type 3A': 'self._PseKRAAC_type_3A()',
        'PseKRAAC type 3B': 'self._PseKRAAC_type_3B()',
        'PseKRAAC type 4': 'self._PseKRAAC_type_4()',
        'PseKRAAC type 5': 'self._PseKRAAC_type_5()',
        'PseKRAAC type 6A': 'self._PseKRAAC_type_6A()',
        'PseKRAAC type 6B': 'self._PseKRAAC_type_6B()',
        'PseKRAAC type 6C': 'self._PseKRAAC_type_6C()',
        'PseKRAAC type 7': 'self._PseKRAAC_type_7()',
        'PseKRAAC type 8': 'self._PseKRAAC_type_8()',
        'PseKRAAC type 9': 'self._PseKRAAC_type_9()',
        'PseKRAAC type 10': 'self._PseKRAAC_type_10()',
        'PseKRAAC type 11': 'self._PseKRAAC_type_11()',
        'PseKRAAC type 12': 'self._PseKRAAC_type_12()',
        'PseKRAAC type 13': 'self._PseKRAAC_type_13()',
        'PseKRAAC type 14': 'self._PseKRAAC_type_14()',
        'PseKRAAC type 15': 'self._PseKRAAC_type_15()',
        'PseKRAAC type 16': 'self._PseKRAAC_type_16()',
        'KNN': 'self._KNN()',
    }
