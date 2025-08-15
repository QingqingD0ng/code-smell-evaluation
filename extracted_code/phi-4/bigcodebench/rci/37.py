import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

def task_func(df, target_column, random_state=42, figsize=(10, 6)):
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
    
    X = df.drop(columns=[target_column])
    if X.empty:
        raise ValueError("No features available for training after dropping the target column.")
    
    y = df[target_column]
    
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X, y)
    
    importance_df = pd.DataFrame({
        'Features': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=figsize)
    sns.barplot(x='Importance', y='Features', data=importance_df)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title('Visualizing Important Features')
    ax = plt.gca()
    
    return model, ax