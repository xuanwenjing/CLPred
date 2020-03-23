# CLPred
A Deep Learning Framework for sequence-based Protein Crystallization Prediction
# Abstract

### Motivation
Determining the structures of proteins is a critical step to fully understand their biological functions. Crystallography-based X-ray diffraction technique is the main method for experimental protein structure determination. However, the underlying crystallization process, which needs multiple costly experimental steps, has a high attrition rate. To overcome this issue, a series ofin-silico methods have been developed with the primary aim of selecting the protein sequences that are potentially promising to be crystallized. However, the predictive performance of the current methods is modest. 

### Results
We propose a deep learning model, so-called CLPred, which uses a bidirectional recurrent neural network with long short-term memory (BLSTM) to capture the long-range interaction patterns between k-mers amino acids. Using sequence only information, CLPred outperforms the existing deep-learning predictors and a vast majority of sequence-based diffraction-quality crystals predictors on three independent test sets. The results highlight the effectiveness of BLSTM in capturing non-local, long-range inter-peptide interaction patterns to distinguish proteins that can result in diffraction-quality crystals from those that cannot. CLPred has been steadily improved over the previous window-based neural networks, which is able to predict crystallization propensity with higher accuracy.

![draft](https://github.com/xuanwenjing/CLPred/blob/master/20200323114312.png)

# Installation

### Requirements

This step will install all the dependencies required for running CLPred. You do not need sudo permissions for this step.

  - Install Anaconda
    1. Download Anaconda (64 bit) installer python3.x for linux : https://www.anaconda.com/distribution/#download-section
    2. Run the installer : `bash Anaconda3-2019.03-Linux-x86_64.sh` and follow the instructions to install.
    3. Install xgboost: conda install -c conda-forge xgboost 
    4. Install shap: conda install -c conda-forge shap 
    5. Install Bio: conda install -c anaconda biopython 
    
# Run CLPred in Train Mode

To run BCrystal for training xgboost model on our training proteins, you need to do the following:

### Execute in the command line
  1. `Rscript --vanilla features_PaRSnIP_v2.R Data/Train/FULL_Train.fasta`
  2. `python xgb_train.py`
  
 # Run CLPred on New Test file

### Execute in the command line
 
  1. `Rscript --vanilla features_PaRSnIP_v2.R <your-test>.fasta`
  2. `python xgb.py features.csv <your-test>.fasta <output_folder>`
