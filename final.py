import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier, Pool
import optuna
from tqdm import tqdm

# Загрузка данных
train = pd.read_parquet('merged_dataframes/train.parquet')
val = pd.read_parquet('merged_dataframes/val.parquet')
test = pd.read_parquet('merged_dataframes/test.parquet')

# Подготовка данных
common_columns = list(set(train.columns) & set(val.columns) & set(test.columns))
train = train[common_columns + ['is_double']]
val = val[common_columns + ['is_double']]
test = test[common_columns]

numeric_features = train.select_dtypes(include=[np.number]).columns.difference(['is_double']).tolist()
mean_values = train[numeric_features].mean()

for df in [train, val, test]:
    df[numeric_features] = df[numeric_features].fillna(mean_values)
    df[numeric_features] = df[numeric_features].replace([np.inf, -np.inf], 0)

scaler = StandardScaler()
train[numeric_features] = scaler.fit_transform(train[numeric_features])
val[numeric_features] = scaler.transform(val[numeric_features])
test[numeric_features] = scaler.transform(test[numeric_features])

cat_features = list(set(train.columns) - set(numeric_features) - {'is_double'})
for df in [train, val, test]:
    for col in cat_features:
        df[col] = df[col].fillna('unknown')

X_train = train[numeric_features + cat_features]
y_train = train['is_double']
X_val = val[numeric_features + cat_features]
y_val = val['is_double']
X_test = test[numeric_features + cat_features]

# Optuna objective
def objective(trial):
    bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"])
    params = {
        "iterations": trial.suggest_int("iterations", 300, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
        "bootstrap_type": bootstrap_type,
        "random_strength": trial.suggest_float("random_strength", 0.1, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0)
                               if bootstrap_type == "Bayesian" else None,
        "od_type": "Iter",
        "od_wait": trial.suggest_int("od_wait", 20, 50),
        "random_seed": 42,
        "verbose": 0,
        "task_type": "GPU",
    }
    params = {k: v for k, v in params.items() if v is not None}
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, cat_features=cat_features)
    preds = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, preds)

# Подбор с сохранением лучших моделей
study = optuna.create_study(direction="maximize")
N_TRIALS = 30
os.makedirs("best_models", exist_ok=True)
best_score = -1

for i in tqdm(range(N_TRIALS), desc="Optuna trials"):
    study.optimize(objective, n_trials=1, catch=(Exception,))
    current_score = study.best_value
    if current_score > best_score:
        best_score = current_score
        best_params = study.best_params

        # Обучаем и сохраняем новую лучшую модель
        best_model = CatBoostClassifier(**best_params, random_seed=42, verbose=0, cat_features=cat_features, task_type='GPU')
        best_model.fit(Pool(X_train, y_train, cat_features=cat_features))

        model_path = f"best_models/model_trial_{i+1}_score_{best_score:.4f}.cbm"
        best_model.save_model(model_path)
        print(f"✔ Saved new best model to {model_path} with score = {best_score:.4f}")

# Финальная модель из лучших параметров
final_model = CatBoostClassifier(**study.best_params, random_seed=42, verbose=100, cat_features=cat_features, task_type='GPU')
final_model.fit(Pool(X_train, y_train, cat_features=cat_features))

# Сохранение важности признаков
feat_imp = pd.DataFrame({
    "feature": X_train.columns,
    "importance": final_model.get_feature_importance(type='PredictionValuesChange')
}).sort_values("importance", ascending=False)
feat_imp.to_csv("feature_importance.csv", index=False)

# Предсказания на тесте
preds_test = final_model.predict_proba(X_test)[:, 1]
pd.DataFrame({ "prediction": preds_test }).to_csv("test_predictions.csv", index=False)
