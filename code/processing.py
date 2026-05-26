import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

class DataPipeline:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.selected_features = None

    def load_and_analyze(self):
        """Loads data and prints summary statistics before cleaning (replicates Analysis.java)"""
        self.df = pd.read_csv(self.filepath)
        print("\n=== Analysis Before Cleaning ===")
        print(f"Number of attributes: {self.df.shape[1]}")
        print(f"Number of instances: {self.df.shape[0]}")
        print("\nMissing values per attribute:")
        print(self.df.isnull().sum())
        return self.df

    def clean_data(self):
        """Handles missing values, drops duplicates, and runs IQR outlier handling (HandleDirty.java)"""
        print("\n=== Executing Data Cleaning Pipeline ===")
        
        # Handle Missing Values: Impute numeric with median, nominal with mode (ReplaceMissingValues equivalent)
        for col in self.df.columns:
            if self.df[col].dtype in [np.float64, np.int64]:
                self.df[col] = self.df[col].fillna(self.df[col].median())
            else:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        print("Missing values handled.")

        # Remove Duplicates
        initial_count = len(self.df)
        self.df.drop_duplicates(inplace=True)
        print(f"Duplicates removed. Dropped {initial_count - len(self.df)} identical rows.")

        # Outlier Mitigation via IQR (replicates the threshold ratio logic from HandleDirty)
        threshold = 0.01
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_ratio = len(outliers) / len(self.df)
            
            print(f"Attribute {col}: Outliers = {len(outliers)} ({outlier_ratio:.2%})")
            if outlier_ratio > threshold:
                print(f"  → Ratio > threshold ({threshold}). Replacing outliers with column median...")
                median_val = self.df[col].median()
                self.df.loc[(self.df[col] < lower_bound) | (self.df[col] > upper_bound), col] = median_val

        return self.df

    def select_features(self, target_column, num_to_select=11):
        """Applies Information Gain (Mutual Information) ranking to select top features"""
        print(f"\n=== Applying Feature Selection (Information Gain Ranker) ===")
        
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        
        # Encode temporary object categories to compute Mutual Information numerical scores safely
        X_encoded = X.copy()
        for col in X_encoded.select_dtypes(include=['object', 'string']).columns:
            X_encoded[col] = X_encoded[col].astype('category').cat.codes
            
        scores = mutual_info_classif(X_encoded, y, random_state=1)
        mi_series = pd.Series(scores, index=X.columns).sort_values(ascending=False)
        
        print("\nInformation Gain Scores:")
        for col, score in mi_series.items():
            print(f"{col} : {score:.5f}")
            
        self.selected_features = list(mi_series.head(num_to_select).index)
        print(f"\nSelected top {num_to_select} attributes: {self.selected_features}")
        
        # Keep selected features + the target label
        self.df = self.df[self.selected_features + [target_column]]
        return self.df
