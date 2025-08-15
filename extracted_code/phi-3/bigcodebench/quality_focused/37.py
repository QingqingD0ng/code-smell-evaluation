import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

def task_func(df, target_column):
    model = RandomForestClassifier(random_state=42)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    model.fit(X, y)
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    sorted_importances = feature_importances.sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_importances, y=sorted_importances.index)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title('Visualizing Important Features')
    plt.tight_layout()
    return model, plt.gcf()

# Example usage:
# data = pd.DataFrame({"X": [-1, 3, 5, -4, 7, 2], "label": [0, 1, 1, 0, 1, 1]})
# model, ax = task_func(data, "label")
# print(data.head(2))
# print(model)