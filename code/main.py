import os
from processing import DataPipeline
from classification import ClassificationEngine
from association import AssociationMiner

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    input_csv = os.path.join(root_dir, "data", "heart_disease.csv") 
    output_dir = os.path.join(root_dir, "data")
    cleaned_csv_output = os.path.join(output_dir, "After_Cleaning.csv")
    target_variable = "Heart Disease Status"
    if not os.path.exists(input_csv):
        print(f"Error: Cannot find the data file at: {input_csv}")
        print("Please check your folder structure and try again.")
        return
    print(f"Found dataset successfully at: {input_csv}")

    # Ingestion and Cleaning Pipelines
    pipeline = DataPipeline(input_csv)
    pipeline.load_and_analyze()
    pipeline.clean_data()
    
    # Feature Selection Configuration (Top 11 Informative Features)
    selected_df = pipeline.select_features(target_column=target_variable, num_to_select=11)
    
    # Save processed dataset copy for auditing trail back into the data folder
    selected_df.to_csv(cleaned_csv_output, index=False)
    print(f"Cleaned dataset saved safely to: {cleaned_csv_output}")
    
    # Train-Holdout and K-Fold CV Classification Architecture
    engine = ClassificationEngine(selected_df, target_variable)
    X_train, X_test, y_train, y_test = engine.execute_train_test_split()
    engine.run_direct_evaluation(X_train, X_test, y_train, y_test, output_dir=output_dir)
    engine.run_robust_cross_validation()
    
    # Association Mining and Unsupervised Rule Mining
    miner = AssociationMiner(selected_df)
    miner.discretize_and_mine(target_col=target_variable, bins=5, min_support=0.05, min_threshold=0.8)

if __name__ == "__main__":
    main()