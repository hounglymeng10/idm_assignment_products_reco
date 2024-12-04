# Importing libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Creating the dataset
data = {
    'Bread': [1, 1, 0, 1, 1],
    'Milk': [1, 0, 1, 1, 1],
    'Diaper': [0, 1, 1, 1, 0],
    'Beer': [0, 1, 1, 1, 0],
    'Coke': [0, 0, 1, 0, 1],
    'Eggs': [0, 1, 0, 0, 0]
}

# Converting to a DataFrame
df = pd.DataFrame(data)
print("Dataset:")
print(df)

# Applying Apriori to find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
print("\nFrequent Itemsets:")
print(frequent_itemsets)

# Generating association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
