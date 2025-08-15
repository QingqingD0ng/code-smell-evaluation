import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

def task_func(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    feature_importances = model.feature_importances_
    sorted_idx = feature_importances.argsort()[::-1]

    plt.figure()
    sns.barplot(x=feature_importances[sorted_idx], y=X.columns[sorted_idx])
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title('Visualizing Important Features')
    plt.show()

    return model, plt

# Example usage:
# data = pd.DataFrame({"X": [-1, 3, 5, -4, 7, 2], "label": [0, 1, 1, 0, 1, 1]})
# model, ax = task_func(data, "label")
# print(data.head(2))
# print(model)