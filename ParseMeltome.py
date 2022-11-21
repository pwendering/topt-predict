import pandas as pd
import numpy as np


class ParseMeltome:

    def __init__(self, meltome):
        self.meltome = meltome

    def parse_meltome(self):
        """
        This function parses meltome atlas CSV into a dataframe that contains all proteins as rows and the fractions of
        native protein at different temperatures as columns.
        :return: creates a pandas dataframe and writes it to "cross-species-parsed.csv"
        """

        # read meltome file into pandas dataframe
        df = pd.read_csv(self.meltome, sep=",")
        df.sort_values('Protein_ID', inplace=True)

        # find the unique set of protein IDs - if protein ID value is nan, take gene name instead
        prot_ids = df.Protein_ID.copy()
        gene_names = df.gene_name.copy()
        na_idx = prot_ids.isna()
        prot_ids[na_idx] = gene_names[na_idx]

        # get unique set of protein IDs / gene names
        prot_ids_uniq = prot_ids.unique()
        del prot_ids, gene_names

        # find the unique set of temperatures
        temperatures_uniq = df.temperature.unique()

        # initialize matrix of response variables with numbers of unique genes and temperatures
        y_matrix = np.empty((len(prot_ids_uniq), len(temperatures_uniq)))
        y_matrix.fill(np.nan)

        for i in range(0, len(prot_ids_uniq)):

            # find all rows that correspond to current protein ID
            tmp_df = df.loc[df['Protein_ID'] == prot_ids_uniq[i]].copy()
            tmp_temperatures = tmp_df['temperature']

            col_idx = [np.where(temperatures_uniq == x) for x in tmp_temperatures]
            col_idx = [x for xss in col_idx for xs in xss for x in xs]

            # add fold change value to matrix
            y_matrix[i, col_idx] = tmp_df['fold_change']

            if i % 1000 == 0:
                print('Processed ' + str(i) + ' proteins (' + str(round(100*i/len(prot_ids_uniq), 2)) + '%)')

        # create new dataframe from matrix and write results to a new CSV file
        new_df = pd.DataFrame(y_matrix, columns=temperatures_uniq,
                              index=pd.Index(prot_ids_uniq, name='Protein_ID'))
        print(new_df.head())

        new_df.to_csv('cross-species-parsed.csv', lineterminator='\n')
