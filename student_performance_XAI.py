import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE


def load_data(file_path="student-por.csv"):
    df = pd.read_csv(file_path, encoding="utf-8")
    df.columns = df.columns.str.strip()

    df["pass"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)
    df.drop(["G1", "G2", "G3"], axis=1, inplace=True)

    return df


def feature_engineering(df):
    df["traveltime"] = df["traveltime"].replace(0, 1)

    df["study_efficiency"] = df["studytime"] / (df["traveltime"] + 1)
    df["parent_edu_avg"] = (df["Medu"] + df["Fedu"]) / 2

    support_columns = ["schoolsup", "famsup", "paid", "activities"]

    for col in support_columns:
        if df[col].dtype == "object":
            df[col] = df[col].map({"yes": 1, "no": 0})

    df["support_score"] = df[support_columns].sum(axis=1)
    df["alcohol_score"] = df["Dalc"] + df["Walc"]
    df["stress_level"] = df["studytime"] + df["traveltime"] + df["absences"]
    df["commitment_ratio"] = df["studytime"] / (df["freetime"] + 1)
    df["high_risk"] = ((df["absences"] > 10) & (df["failures"] > 1)).astype(int)

    return df


def prepare_data(df):
    df_encoded = pd.get_dummies(df, drop_first=True)

    X = df_encoded.drop("pass", axis=1)
    y = df_encoded["pass"]

    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))

    print(f"{model_name} Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print(f"{model_name} ROC-AUC Score: {auc:.4f}")

        fpr, tpr, _ = roc_curve(y_test, y_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"{model_name} AUC = {auc:.2f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend()
        plt.grid(True)
        plt.savefig("roc_curve.png")
        plt.show()

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "F1": f1,
        "Recall": recall,
        "Precision": precision
    }


def train_models(X_train, X_test, y_train, y_test):
    results = []

    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            random_state=42
        ),
        "Random Forest SMOTE": RandomForestClassifier(random_state=42),
        "SVM": SVC(C=10, kernel="rbf", gamma="scale", probability=True),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(50, 50),
            activation="tanh",
            learning_rate_init=0.01,
            max_iter=300,
            random_state=42
        )
    }

    for name, model in models.items():
        if name == "Random Forest SMOTE":
            model.fit(X_train_sm, y_train_sm)
        else:
            model.fit(X_train, y_train)

        result = evaluate_model(model, X_test, y_test, name)
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv("model_comparison.csv", index=False)

    print("\nModel Comparison:")
    print(results_df)

    return results_df


def main():
    df = load_data()
    df = feature_engineering(df)

    X_train, X_test, y_train, y_test, feature_names = prepare_data(df)

    train_models(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
