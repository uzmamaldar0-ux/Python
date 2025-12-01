# Python"""
Apriori Algorithm – Market Basket Analysis
"""
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("association_rules_transactions_5000.csv")
transactions = df["items"].apply(lambda x: x.split(", ")).tolist()

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
data = pd.DataFrame(te_array, columns=te.columns_)

freq_items = apriori(data, min_support=0.05, use_colnames=True)
print("Frequent Itemsets:")
print(freq_items)

rules = association_rules(freq_items, metric="lift", min_threshold=1.0)
print("\nAssociation Rules:")
print(rules.sort_values(by="lift", ascending=False).head(10))
