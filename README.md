# Prediction of optimal temperature of protein stability using sequence features

## Requirements

* Python 3.9
* OS: tested on Windows 10 and Ubuntu 20.04.3 LTS

## Setup

1) Clone the GitHub repository
`$ git clone https://github.com/pwendering/topt-predict`
2) Create a virtual environment "topt-predict" from the requirements.txt file

```
$ cd toptpredict
$ virtualenv topt-predict
$ source topt-predict/bin/activate
$ pip install -r requirements.txt
```

If you don't have virtualenv installed, you can get it using `pip install virtualenv` on the command line.

The trained model file is not included in the repository because of its large size.
It can be created by running
```
python3.9 create_rf_model.py
```
This script also extracts feature importances and runs a cross-validation scheme to evaluate the performance.

## Usage of predict_topt.py

Use the `-h` option to display required and optional input arguments:

```
python3.9 predict_topt.py -h

usage: predict_topt [-h] [-m MODEL] [-e] [-p] seq_file

Predition of optimal protein temperatures based on the amino acid sequence

positional arguments:
  seq_file              fasta-formatted file containing amino acid sequences

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        regression model file (.sav)
  -e, --extract_features
                        extract features from amino acid sequences (will be stored in same directory as sequence file)
  -p, --predict         predict key temperatures
  
```

The amino acid sequence file should not contain newlines in the amino acid sequences and look this like this:

```
>ID1
MAIAFKSGVFFLQSPKSQIGFRHSSPPD...
>ID2
MRSLLHRTILLTSPSHSLIRRTSLSAMA...
...
```

### Extract features

By using the `-e` option, all required features for the prection of T<sub>opt</sub> will be extracted:

`python3.9 -e predict_topt.py aa_seqs.fasta`.

Specifically, the extracted features are:
* AAC (Amino acid composition)
* QSOrder (Quasi-sequence-order descriptors)
* CTDC (Composition)
* PAAC (Pseudo-amino acid composition)
* AC (Auto covariance)
* CTriad (Conjoint triad)
* DistancePair (PseAAC of distance-pairs and reduced alphabet)
* GAAC (Grouped amino acid composition)
* Geary (Geary)
* Moran (Moran)
* SOCNumber (Sequence-order-coupling number)
* CKSAAP type 1 (Composition of k-spaced amino acid pairs type 1 - normalized)
* CTDD (Distribution)
* DPC type 1 (Dipeptide composition type 1 - normalized)
* GDPC type 1 (Grouped dipeptide composition type 1 - normalized)
* NMBroto (Normalized Moreau-Broto)
* PseKRAAC type 10 (Pseudo K-tuple reduced amino acids composition type 10)

For more info on the features above, please refer to the [iFeatureOmega-CLI](https://github.com/Superzchen/iFeatureOmega-CLI) repository.

* SEQL (sequence length)
* MW (molecular weight)
* ISOEP (isoelectric point)
* AROM (aromaticity)
* INST (instability index)
* HELIXF (fraction of helix in secondary structure)
* TURNF (fraction of turn in secondary structure)
* SHEETF (fraction of sheet in secondary structure)
* REMAINF (fraction of unaccounted secondary structure elements)
* MEXTC_1 (molar extinction coefficient)
* GRAVY (gravy according to Kyte and Doolittle)

For more info on the features above, please refer to the documentation of the biopython module [ProtParam](https://biopython.org/docs/1.75/api/Bio.SeqUtils.ProtParam.html).

* ENTROPY (entropy, for more info please refer to the documentation of the [protlearn package](https://protlearn.readthedocs.io/en/latest/introduction.html)

* G1 (hydrophobic amino acids: cumulative fraction of V, I, L, F, M, W, Y, and C)
* G2 (negatively charged amino acids: cumulative fraction of D and E)
* G3 (positively charged amino acids: cumulative fraction of R, K, and H)
* G4 (conformational amino acids: cumulative fraction of G and P)
* G5 (polar amino acids: cumulative fraction of N, Q, and S)

The definition of amino acid groups was taken from the paper of Yang et al. (2022) Int. J. Mol. Sci. 2022, 23(18), 10798 (DOI:[10.3390/ijms231810798](https://doi.org/10.3390/ijms231810798)).

### Predict optimal temperatures

To extract features and predict stability optima in one go, run `python3.9 -e -p predict_topt.py aa_seqs.fasta`.

All features will be standardized prior to the prediction, using the distribtion of the training data (Meltome Atlas, published by Jarząb et al. (2020) Nat. Methods 17, 495–503 (DOI:[10.1038/s41592-020-0801-4](https://doi.org/10.1038/s41592-020-0801-4)).
