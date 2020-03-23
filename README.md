# CLPred
A Deep Learning Framework for sequence-based Protein Crystallization Prediction
# Abstract

### Motivation
Determining the structures of proteins is a critical step to fully understand their biological functions. Crystallography-based X-ray diffraction technique is the main method for experimental protein structure determination. However, the underlying crystallization process, which needs multiple costly experimental steps, has a high attrition rate. To overcome this issue, a series ofin-silico methods have been developed with the primary aim of selecting the protein sequences that are potentially promising to be crystallized. However, the predictive performance of the current methods is modest. 

### Results
We propose a deep learning model, so-called CLPred, which uses a bidirectional recurrent neural network with long short-term memory (BLSTM) to capture the long-range interaction patterns between k-mers amino acids. Using sequence only information, CLPred outperforms the existing deep-learning predictors and a vast majority of sequence-based diffraction-quality crystals predictors on three independent test sets. The results highlight the effectiveness of BLSTM in capturing non-local, long-range inter-peptide interaction patterns to distinguish proteins that can result in diffraction-quality crystals from those that cannot. CLPred has been steadily improved over the previous window-based neural networks, which is able to predict crystallization propensity with higher accuracy.


# Installation

### Requirements

This step will install all the dependencies required for running CLPred. You do not need sudo permissions for this step.

  - Install Anaconda
    1. Download Anaconda (64 bit) installer python3.x for linux : https://www.anaconda.com/distribution/#download-section
    2. Run the installer : `bash Anaconda3-2019.03-Linux-x86_64.sh` and follow the instructions to install.
    3. Install tensorflow-gpu: conda install tensorflow-gpu 
    4. Install imblearn: conda install -c glemaitre imbalanced-learn
    5. Install sklearn: conda install -c conda-forge scikit-learn
    
# Run CLPred in Train Mode
  `CUDA_VISIBLE_DEVICES=? python Train.py`
  
# Run CLPred on New Test file
  `CUDA_VISIBLE_DEVICES=? python Test.py'

You can change the input file in the code.
If you have any questions, please contact xuanwj@csu.edu.cn
