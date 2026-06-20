# 🧠 NIDS with Genetic Algorithm and MLP

This repository contains the implementation of a Network Intrusion Detection System (NIDS) project. The core idea is to leverage a **Genetic Algorithm (GA)** for efficient **feature selection** and then use a **Multi-Layer Perceptron (MLP)** neural network for classifying network traffic as either normal or an attack.

---

## 🎯 Project Goals

*   **Reduce Dimensionality:** Significantly decrease the number of features in the dataset.
*   **Improve Classification Accuracy:** Maintain or enhance the accuracy of the intrusion detection model.
*   **Optimize Training Time:** Achieve faster model training due to reduced feature set.

---

## 📊 Dataset

The project utilizes the **KDD Cup 1999 (KDD)** dataset, a standard benchmark for evaluating NIDS. After preprocessing, the dataset initially had **122 features**.

---

## 🛠️ Project Pipeline & Key Components

The project is divided into several Python scripts:

1.  **`PreProcessing.py`**:
    *   Loads the KDD dataset (`KDDTest+.txt`).
    *   Handles missing values and irrelevant columns (`difficulty_level`).
    *   Performs **One-Hot Encoding** on categorical features (`protocol_type`, `service`, `flag`), increasing the feature count.
    *   Splits data into training (80%) and testing (20%) sets.
    *   Normalizes numerical features using `MinMaxScaler`.
    *   Saves processed data and the scaler (`.npy`, `.pkl` files).

2.  **`genetic_algorithm.py`**:
    *   Implements a **Genetic Algorithm** for feature selection.
    *   **Chromosome Representation:** Binary strings (1 = select feature, 0 = discard).
    *   **Fitness Function:** Balances classification accuracy and feature reduction ratio:
        $$Fitness = 0.9 \times Accuracy + 0.1 \times \left(1.0 - \frac{N_{selected}}{N_{total}}\right)$$
    *   **GA Operators:** Roulette wheel selection, single-point crossover, bit-flip mutation (rate 0.01).
    *   **Evaluation Model:** A small MLP is used internally within GA to calculate fitness.
    *   **Elitism:** Applied to preserve the best individuals.
    *   Saves the best selected features (`best_genetic_features.npy`).

3.  **`baseline_model.py`**:
    *   Trains a **baseline MLP model** using **all 122 features** from the preprocessed data.
    *   Used for comparison with the GA-optimized model.

4.  **`Final_model.py`**:
    *   Trains the **final MLP model** using **only the features selected by the Genetic Algorithm** (`best_genetic_features.npy`).
    *   Features a deeper MLP architecture (4 hidden layers).

---

## 📈 Results

| Metric              | Baseline Model (122 Features) | GA-MLP Model (53 Features) | Improvement |
| :------------------ | :---------------------------- | :------------------------- | :---------- |
| **Features Count**  | 122                           | **53**                     | **-56.55%** |
| **Overall Accuracy**| 96.45%                        | **96.82%**                 | **+0.37%**  |
| **Precision (Class 0 - Normal)** | 0.96                          | **0.97**                   | **+0.01**   |
| **Precision (Class 1 - Attack)** | 0.97                          | 0.97                       | -           |
| **Training Time**   | Full 500 epochs               | Early Stopping             | Faster      |

---

## 💡 Conclusion

The Genetic Algorithm effectively reduced the feature space by **56.55%** (from 122 to 53 features). Importantly, this dimensionality reduction not only maintained but slightly improved the overall accuracy (from 96.45% to 96.82%) and led to a significant reduction in training time due to early stopping. This highlights the effectiveness of GA-based feature selection in enhancing NIDS performance, especially with high-dimensional datasets.

---

## 🚀 Getting Started (How to Run)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/NIDS-GA-MLP.git
    cd NIDS-GA-MLP
    ```
    *(Replace `YOUR_USERNAME` with your GitHub username)*

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Make sure you have `numpy`, `pandas`, `scikit-learn`, `joblib` installed)*

3.  **Run the scripts (example):**
    *   To preprocess data: `python PreProcessing.py`
    *   To run GA feature selection: `python genetic_algorithm.py`
    *   To train the baseline model: `python baseline_model.py`
    *   To train the final GA-MLP model: `python Final_model.py`

*(Note: Ensure you have the KDD dataset file (`KDDTest+.txt`) in the same directory or adjust the file path in `PreProcessing.py`)*

---

## 📚 Technologies Used

*   Python
*   Scikit-learn
*   NumPy
*   Pandas
*   Joblib (for saving/loading models and scaler)

---

## 📝 Author

*   [SamanFardin] - [https://www.linkedin.com/in/SamanFardin/]
