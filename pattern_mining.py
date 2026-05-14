import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import time

def load_transactions(file_path='transactions.csv'):
    df = pd.read_csv(file_path)
    # Convert back to list of lists
    transactions = df['Transaction_Str'].apply(lambda x: x.split(',')).tolist()
    return transactions

def run_pattern_mining(transactions, min_support=0.05):
    print(f"Number of transactions: {len(transactions)}")
    
    # One-hot encode transactions for mlxtend
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    results = {}
    
    # 1. Apriori
    print(f"\n--- Running Apriori (min_support={min_support}) ---")
    start_time = time.time()
    apriori_frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    apriori_time = time.time() - start_time
    
    print(f"Apriori Time: {apriori_time:.4f} seconds")
    print(f"Apriori Found {len(apriori_frequent_itemsets)} itemsets")
    results['Apriori'] = {
        'time': apriori_time,
        'itemsets': apriori_frequent_itemsets
    }
    
    # 2. FP-Growth
    print(f"\n--- Running FP-Growth (min_support={min_support}) ---")
    start_time = time.time()
    fp_frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
    fp_time = time.time() - start_time
    
    print(f"FP-Growth Time: {fp_time:.4f} seconds")
    print(f"FP-Growth Found {len(fp_frequent_itemsets)} itemsets")
    results['FP-Growth'] = {
        'time': fp_time,
        'itemsets': fp_frequent_itemsets
    }
    
    # Association Rules (using FP-growth output as they are identical sets)
    rules = association_rules(fp_frequent_itemsets, metric="confidence", min_threshold=0.5, num_itemsets=len(fp_frequent_itemsets))
    # Sort by confidence and lift
    rules = rules.sort_values(by=['confidence', 'lift'], ascending=[False, False])
    
    return results, rules

if __name__ == '__main__':
    transactions = load_transactions()
    results, rules = run_pattern_mining(transactions, min_support=0.01)
    
    print("\n--- Top Association Rules ---")
    # Clean up the output formatting
    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
    rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10).to_string(index=False))
