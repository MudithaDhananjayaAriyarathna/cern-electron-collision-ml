# Machine Learning for Electron-Collision Events

This project applies machine-learning regression methods to a CERN/CMS electron-collision dataset to predict the **invariant mass of two-electron events**.

The project was completed as part of **Computational Physics Laboratory 2** during my undergraduate studies in Computational Physics at the University of Colombo.

## Project Overview

Modern high-energy-physics experiments produce large datasets that require computational methods for efficient analysis. In this project, a regression approach was used to investigate whether machine-learning models could learn the relationship between measured electron properties and the invariant mass of the two-electron system.

The analysis compares:

* **Random Forest Regression**
* **Deep Neural Network (DNN)**
* **Principal Component Analysis (PCA)** for dimensionality reduction

The models were evaluated using **MAE, MSE, RMSE, and R²** on both training and test data.

## Dataset

The project uses the publicly available **CMS/CERN Open Data — Events with two electrons from 2010** dataset.

The original dataset contains **100,000 collision events** with 19 attributes describing the two electrons and their kinematic properties. The target variable is the invariant mass, `M`, measured in GeV.

The dataset is **not included in this repository**. Please obtain the original dataset from the CERN Open Data Portal and place it at:

```text
data/dielectron.csv
```

### Dataset source

**CERN Open Data:** https://opendata.cern.ch/record/304

**DOI:** `10.7483/OPENDATA.CMS.PCSW.AHVG`

Please refer to the official CERN Open Data record for the dataset's licensing and attribution information.

## Methodology

### 1\. Data preprocessing

The original dataset contains 100,000 entries and 19 attributes.

The preprocessing workflow includes:

* Removing the `Run` and `Event` identifiers
* Checking and removing rows with missing target values
* Removing duplicate entries
* Examining potential outliers using percentile bounds
* Creating composite features:

  * Transverse momentum magnitude
  * Absolute pseudorapidity difference
  * Energy difference
* Examining correlations with the invariant mass
* Removing selected weakly correlated input attributes

After preprocessing, the dataset contains 99,892 entries and 13 attributes including the target.

### 2\. Random Forest Regression

Random Forest regression was selected because it can model nonlinear relationships between the collision variables and invariant mass.

Hyperparameters were optimized using **GridSearchCV with three-fold cross-validation**. The search considered:

* `n\_estimators`
* `max\_depth`
* `min\_samples\_split`
* `min\_samples\_leaf`

Five training subsets were used and evaluated against a common test set.

### 3\. Deep Neural Network

A fully connected neural network was used as a second nonlinear regression model.

The network architecture follows:

```text
Input
  ↓
64 neurons — ReLU
  ↓
64 neurons — ReLU
  ↓
32 neurons — ReLU
  ↓
1 neuron — Linear output
```

The model uses:

* Adam optimizer
* Mean Squared Error (MSE) loss
* Mean Absolute Error (MAE) as an evaluation metric
* Batch size of 64
* Up to 200 epochs
* Early stopping based on validation loss

The input features were standardized before DNN training.

### 4\. Principal Component Analysis

PCA was applied after standardization to investigate the effect of dimensionality reduction.

The analysis retained **95% of the variance**, reducing the original feature space to **7 principal components**.

Both Random Forest and DNN models were then evaluated using the PCA-transformed features.

## Results

The original analysis found that both Random Forest and DNN models achieved strong predictive performance.

For the models trained on the original feature space:

* Random Forest achieved an average test **R² of approximately 0.959**
* DNN achieved an average test **R² of approximately 0.960**

The DNN models showed a smaller train–test performance gap than the Random Forest models, indicating better generalization.

After PCA:

* Random Forest performance decreased, suggesting that some feature-specific information useful to the tree-based model was lost.
* DNN performance remained approximately unchanged while benefiting from reduced dimensionality and faster training.

Overall, the original project concluded that **DNN with PCA provided the most balanced and efficient approach** among the investigated models.

> The numerical results reported here are based on the original project report. Results may vary slightly when the notebook is rerun because of software versions and hardware-dependent behavior in neural-network training.

## Repository Structure

```text
cern-electron-collision-ml/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── notebooks/
│   └── cern\_electron\_collision\_ml.ipynb
│
├── data/
│   └── README.md
│
├── figures/
│
└── report/
    └── project\_report.pdf
```

## Technologies

* **Python**
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow / Keras
* Jupyter Notebook / Google Colab

## Reproducibility

To reproduce the analysis:

1. Clone this repository.
2. Download the original CERN/CMS dataset from the official source.
3. Place the dataset at:

```text
data/dielectron.csv
```

4. Install the required Python packages:

```bash
pip install -r requirements.txt
```

5. Open:

```text
notebooks/cern\_electron\_collision\_ml.ipynb
```

6. Run the notebook from beginning to end.

Random seeds are fixed where practical. Nevertheless, small differences in DNN results may occur depending on the TensorFlow version, hardware, and execution environment.

## Academic Context

**Project:** Computational Physics Laboratory 2 
**Field:** Computational Physics / Machine Learning / High-Energy Physics  
**Programming Language:** Python  
**Methods:** Random Forest Regression, Deep Neural Network, Principal Component Analysis

## Author

**M.M.M.D. Ariyarathna**  
B.Sc. (Hons) in Computational Physics  
University of Colombo, Sri Lanka

