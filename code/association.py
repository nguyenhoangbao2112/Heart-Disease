import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

class AssociationMiner:
    def __init__(self, data):
        self.data = data.copy()

    def discretize_and_mine(self, target_col, bins=5, min_support=0.05, min_threshold=0.8):
        """Converts numeric values to 5 categorical bins, formats transactions via one-hot boolean frames, and extracts Apriori rules"""
        print("\n=== Replicating Association Rule Mining (Apriori) ===")
        
        # Discretize all continuous features into 5 numeric structural ranges
        for col in self.data.columns:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                self.data[col] = pd.cut(self.data[col], bins=bins, labels=[f"Low", f"Low-Mid", f"Medium", f"Mid-High", f"High"])
        
        # Map structural features to separate true/false transaction variables (One-Hot dummy matrix)
        df_encoded = pd.get_dummies(self.data, columns=self.data.columns, dtype=bool)
        
        print("\nProcessing transactional transaction mapping frame shapes...")
        
        # Apply Apriori algorithms
        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
        print(f"Found {len(frequent_itemsets)} frequent itemsets using support >= {min_support}.")
        
        if frequent_itemsets.empty:
            print("No frequent itemsets matched your parameters.")
            return None

        # Filter generation metrics using confidence metrics rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_threshold)
        
        # Filter explicitly where the rules indicate an outcome involving your target class status
        target_rules = rules[rules['consequents'].apply(lambda x: any(target_col in str(item) for item in x))]
        
        print(f"\nTop 10 Mining Association Rules generated (Filtered by Target Label):")
        sorted_rules = target_rules.sort_values(by="lift", ascending=False).head(10)
        
        for idx, row in sorted_rules.iterrows():
            antecedents = list(row['antecedents'])
            consequents = list(row['consequents'])
            print(f"Rule: {antecedents} -> {consequents}")
            print(f"  Support: {row['support']:.4f} | Confidence: {row['confidence']:.4f} | Lift: {row['lift']:.4f}\n")
            
        return sorted_rules