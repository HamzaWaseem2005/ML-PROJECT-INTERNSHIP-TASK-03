
```markdown
# Student Dropout Prediction Project

An end-to-end Machine Learning project designed to analyze, predict, and prevent student dropouts using a structured data pipeline and an interactive web application.

---

## 📂 Project Structure
```text
├── dataset/            # Student Dropout dataset
├── Images/             # EDA (Exploratory Data Analysis) graphs and plots
├── Notebook/           # Google Colab notebooks for training and data analysis
├── Report/             # Documentation and project report files
└── Streamlit/          # Streamlit app code and user interface screenshots

```

---

## 🔍 Project Workflow & Methodology

### 1. Data Preprocessing & Cleaning

* **Handling Missing Values:** Cleaned the dataset to ensure no null values interfered with model performance.
* **Outlier Detection:** Identified and handled extreme outliers to prevent model bias.
* **Correlation Checking:** Analyzed feature correlations to remove multicollinearity and keep only the most impactful variables.
* **Dropping Unnecessary Columns:** Removed redundant or irrelevant columns that did not contribute to the target prediction.

### 2. Feature Engineering

* **Binning:** Grouped continuous data into meaningful categories or bins to improve pattern recognition.
* **New Feature Creation:** Engineered custom features derived from the 36 original attributes to enhance predictive power.

### 3. Pipelines & Transformers

* **Column Transformer:** Used to seamlessly apply distinct preprocessing steps to numerical and categorical features simultaneously.
* **Standard Scaler:** Standardized numerical variables to bring them onto a uniform scale, ensuring optimal performance for distance-based algorithms.

### 4. Model Training & Hyperparameter Tuning

We experimented with **four different machine learning models**:

1. **Logistic Regression (LR)**
2. **K-Nearest Neighbors (KNN)**
3. **Decision Tree (DT)**
4. **Random Forest (RF)**

* **RandomizedSearchCV (RSCV):** Applied cross-validation and hyperparameter tuning to find the optimal configuration for each model, preventing overfitting and maximizing accuracy.

---

## 🚀 Streamlit Web Application

The final optimized model has been integrated into an interactive **Streamlit Web App**.

* **UI Features:** Users (educators or administrators) can input student metrics interactively.
* **Real-time Predictions:** Instantly outputs whether a student is at risk of dropping out.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas & NumPy** (Data Manipulation)
* **Scikit-Learn** (Machine Learning, Pipelines, Column Transformer, RSCV)
* **Matplotlib & Seaborn** (Exploratory Data Analysis / EDA)
* **Streamlit** (Web Application Deployment)

