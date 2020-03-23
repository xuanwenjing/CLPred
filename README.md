# CLPred
A Deep Learning Framework for sequence-based Protein Crystallization Prediction
# Abstract

### Motivation
Determining the structures of proteins is a critical step to fully understand their biological functions. Crystallography-based X-ray diffraction technique is the main method for experimental protein structure determination. However, the underlying crystallization process, which needs multiple costly experimental steps, has a high attrition rate. To overcome this issue, a series ofin-silico methods have been developed with the primary aim of selecting the protein sequences that are potentially promising to be crystallized. However, the predictive performance of the current methods is modest. 

### Results
We propose a deep learning model, so-called CLPred, which uses a bidirectional recurrent neural network with long short-term memory (BLSTM) to capture the long-range interaction patterns between k-mers amino acids. Using sequence only information, CLPred outperforms the existing deep-learning predictors and a vast majority of sequence-based diffraction-quality crystals predictors on three independent test sets. The results highlight the effectiveness of BLSTM in capturing non-local, long-range inter-peptide interaction patterns to distinguish proteins that can result in diffraction-quality crystals from those that cannot. CLPred has been steadily improved over the previous window-based neural networks, which is able to predict crystallization propensity with higher accuracy.

