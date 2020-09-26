from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd


data_set = [['Apple', 'Beer', 'Rice', 'Chicken'],
 ['Apple', 'Beer', 'Rice'],
 ['Apple', 'Beer'],
 ['Apple', 'Bananas'],
 ['Milk', 'Beer', 'Rice', 'Chicken'],
 ['Milk', 'Beer', 'Rice'],
 ['Milk', 'Beer'],
 ['Apple', 'Bananas']]

te = TransactionEncoder()
array = te.fit_transform(data_set)
print(te.columns_)
print(array)
df = pd.DataFrame(array, columns=te.columns_)
frequent_item_set = apriori(df, min_support=0.5, use_colnames=True)
print(frequent_item_set)