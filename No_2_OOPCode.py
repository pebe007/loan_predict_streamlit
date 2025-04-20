'''
[LO 1, LO 2 – 30 Poin] Seluruh proses training dari algoritma machine learning yang terbaik dibubah 
dalam format OOP  
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import zscore
import pickle
import warnings
warnings.filterwarnings("ignore")

class models:
    def __init__(self, df):
        self.df = df.copy()
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None

    def preprocess(self):
        df = self.df

        # Imputasi income
        df['person_income'].fillna(df['person_income'].median(), inplace=True)
        # Normalisasi gender
        df['person_gender'] = df['person_gender'].str.lower().str.replace(' ', '')  # jadi 'female', 'male', 'female' (dari 'fe male')
        
        # Drop duplicate
        df.drop_duplicates(inplace=True)

        # Remove outlier menggunakan Z score
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        z_scores = np.abs(zscore(df[numeric_cols]))
        df = df[(z_scores < 3).all(axis=1)]

        # Label Encoding untuk data kategori
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        # Splitting data
        X = df.drop('loan_status', axis=1)
        y = df['loan_status']

        # Scaling
        X_scaled = self.scaler.fit_transform(X)

        # Train-Test Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        print("✅ Preprocessing selesai.")

    def tuning(self):
        # Random Forest
        param_grid_rf = {
            'n_estimators': [50, 100, 150, 200, 250],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 3, 4, 5, 10],
            'min_samples_leaf': [1, 2, 3, 4, 5]
        }

        rf = RandomForestClassifier(random_state=42)
        grid_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=3,
                               scoring='accuracy', verbose=1, n_jobs=-1)
        grid_rf.fit(self.X_train, self.y_train)
        self.rf_best = grid_rf.best_estimator_
        print("🎯 Random Forest Best Params:", grid_rf.best_params_)

        # XGBoost
        param_grid_xgb = {
            'n_estimators': [50, 100, 150, 200, 250],
            'max_depth': [3, 6, 10, 15],
            'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
            'subsample': [0.75, 0.8, 1.0]
        }

        xgb = XGBClassifier(eval_metric='logloss', random_state=42)
        grid_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv=3,
                                scoring='accuracy', verbose=1, n_jobs=-1)
        grid_xgb.fit(self.X_train, self.y_train)
        self.xgb_best = grid_xgb.best_estimator_
        print("🎯 XGBoost Best Params:", grid_xgb.best_params_)

    def evaluated(self):
        # Evaluate Random Forest
        rf_pred = self.rf_best.predict(self.X_test)
        rf_acc = accuracy_score(self.y_test, rf_pred)
        print("\nPerforma Random Forest:")
        print("Accuracy:", rf_acc)
        print(classification_report(self.y_test, rf_pred))

        # Evaluate XGBoost
        xgb_pred = self.xgb_best.predict(self.X_test)
        xgb_acc = accuracy_score(self.y_test, xgb_pred)
        print("\nPerforma XGBoost:")
        print("Accuracy:", xgb_acc)
        print(classification_report(self.y_test, xgb_pred))

        # Choose best
        if xgb_acc > rf_acc:
            self.best_model = self.xgb_best
            self.best_model_name = "XGBoost"
        else:
            self.best_model = self.rf_best
            self.best_model_name = "Random Forest"

        print(f"\n🏆 Model terbaik: {self.best_model_name} dengan akurasi {max(xgb_acc, rf_acc)}")

    def save_model(self, filename="best_model.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.best_model, f)
        print(f"📦 Model disimpan sebagai {filename}")

df = pd.read_csv("Dataset_A_loan.csv")

# Asumsi df sudah ada
pipeline = models(df)
pipeline.preprocess()
pipeline.tuning()
pipeline.evaluated()
pipeline.save_model("best_model.pkl")
