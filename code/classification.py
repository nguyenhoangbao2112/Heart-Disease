import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

class ClassificationEngine:
    def __init__(self, data, target_col):
        self.data = data.copy()
        self.target_col = target_col
        
        # Isolate X and y
        self.X = self.data.drop(columns=[target_col])
        self.y = self.data[target_col]
        
        # Dynamically set up Preprocessing Transformer for Categories (OneHot Encoding)
        categorical_cols = self.X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numeric_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ])

    def execute_train_test_split(self):
        """Splits data 70/30 stratified, applies Resample balancing exclusively on train data"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.30, random_state=1, stratify=self.y
        )
        
        # Replicates Weka's biased Resample filter on the training set
        sampler = RandomOverSampler(sampling_strategy='auto', random_state=1)
        
        # Transform categories dynamically for proper array handling
        X_train_trans = self.preprocessor.fit_transform(X_train)
        X_test_trans = self.preprocessor.transform(X_test)
        
        X_train_bal, y_train_bal = sampler.fit_resample(X_train_trans, y_train)
        
        print("\n=== Data Splits and Resampling Distributions ===")
        print(f"Training Set Size: {X_train.shape[0]} | Test Set Size: {X_test.shape[0]}")
        print(f"Original Balanced Train Class Dist: \n{y_train.value_counts()}")
        print(f"Resampled/Balanced Train Class Dist: \n{y_train_bal.value_counts()}")
        
        return X_train_bal, X_test_trans, y_train_bal, y_test

    def run_direct_evaluation(self, X_train, X_test, y_train, y_test, output_dir="output"):
        """Trains individual classifiers on training set and validates against the 30% holdout test set"""
        os.makedirs(output_dir, exist_ok=True)
        
        models = {
            "Naive Bayes": GaussianNB(),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=1)
        }
        
        for name, clf in models.items():
            print(f"\n==================================\nTraining Holdout Evaluation: {name}")
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            
            print(f"\nSummary Metrics Report ({name}):")
            print(classification_report(y_test, preds))
            print(f"Confusion Matrix ({name}):")
            print(confusion_matrix(y_test, preds))

    def run_robust_cross_validation(self):
        """Executes full 10-Fold Stratified Cross Validation preventing Data Leakage using imblearn pipelines"""
        print("\n==================================\nExecuting 10-Fold Cross-Validation Pipeline")
        
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        
        models = {
            "Naive Bayes": GaussianNB(),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=1)
        }
        
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        for name, clf in models.items():
            # Build an integrated pipeline running Resample filtering *only* on active training folds
            pipeline = ImbPipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('resample', RandomOverSampler(random_state=1)),
                ('classifier', clf)
            ])
            
            scores = cross_validate(pipeline, self.X, self.y, cv=cv, scoring=scoring, return_train_score=False)
            print(f"\n--- 10-Fold CV Metrics Summary for {name} ---")
            print(f"Mean CV Accuracy      : {scores['test_accuracy'].mean():.2%}")
            print(f"Mean CV Macro-F1 Score: {scores['test_f1_macro'].mean():.4f}")
            print(f"Mean CV Macro-Recall  : {scores['test_recall_macro'].mean():.4f}")