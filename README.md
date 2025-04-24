# 🚢 Titanic Survival Prediction - ML Project

This project uses machine learning to predict whether a passenger survived the Titanic disaster, based on features such as age, gender, ticket class, fare, and more.

## 📁 Dataset

The dataset used is from [Kaggle's Titanic Competition](https://www.kaggle.com/c/titanic/data).  
Place the `train.csv` file inside the `data/` directory.

## 📊 Features Used

- Pclass (Ticket class)
- Sex
- Age
- SibSp (Siblings/Spouses aboard)
- Parch (Parents/Children aboard)
- Fare
- Embarked (Port of Embarkation)

## 🔧 Preprocessing Steps

- Dropped unnecessary columns: `PassengerId`, `Name`, `Ticket`, `Cabin`
- Filled missing values in `Age`, `Fare`, and `Embarked`
- Encoded categorical variables: `Sex`, `Embarked`
- Scaled numerical features: `Age`, `Fare`

## 🤖 Model

- **Algorithm**: Random Forest Classifier  
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix  
- Trained with 80% of the dataset and tested on 20%

## 📈 Results

Check the `outputs/` folder for:
- `metrics.txt`: Accuracy and classification report
- `confusion_matrix.png`: Visual confusion matrix
- `model.pkl`: Saved trained model

## 🚀 How to Run

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the model training script:
    ```bash
    python titanic_model.py
    ```

## 📂 Project Structure

