import pandas as pd
csv=pd.read_csv("laptop.csv")
print(csv)
noduplicate=csv.drop_duplicates()
print(noduplicate)
