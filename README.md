# BERT for Side-Channel Analysis on ASCAD
**Exploring Transformer Models for Power Trace Analysis**

This repository contains the full implementation of my Master's Thesis:  
**_"Exploring the Performance of BERT Transformers on ASCAD Variable Datasets"_**.  
The project investigates the applicability of Transformer-based architectures, specifically BERT, for side-channel attacks (SCAs) against AES.

---

## ✨ Features
- ✅ **BERT-based architecture for SCA**: fusion of power traces with plaintext embeddings.  
- ✅ **ASCAD variable & fixed key datasets**: works with desync=0, 50, 100.  
- ✅ **Key rank evaluation**: compute and visualize rank evolution across attack traces.  
- ✅ **W&B logging**: monitor training/evaluation live.  
- ✅ **Multiple plaintext embeddings**: hex, byte, ascii.  
- ✅ **Reproducibility**: all code aligned with thesis experiments.  

---

## 📂 Repository Structure
ASCAD_all_latest/
│── train.py # Training pipeline
│── evaluate.py # Evaluation with key rank metrics
│── models/ # BERT model architectures
│── utils/ # Preprocessing, dataset loaders, metrics
│── configs/ # Example experiment configs
│── results/ # Plots, rank evolution results


## 📊 Datasets
This repo is built for **ASCAD v1** datasets by ANSSI:

- `ascad-variable.h5` (desync=0)  
- `ascad-variable_desync50.h5` (desync=50)  
- `ascad-variable_desync100.h5` (desync=100)  
- *(optional)* `ascad-fixed.h5

⚠️ Due to licensing, datasets are not included. Please download them from the [official ASCAD repository](https://github.com/ANSSI-FR/ASCAD).

---

## 🚀 Quick Start1. 
**Clone the repository**
   ```bash
   git clone https://github.com/AliAlperSakar/bert_ascad_sca_masterthesis.git
   cd bert_ascad_sca_masterthesis/work/BERT/ASCAD_all_latest


📬 Contact

Ali Alper Sakar
LinkedIn
