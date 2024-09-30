#import pandas as pd
import csv
with open("laptop.csv",'r') as file:
    csvreader=csv.reader(file)
    for row in csvreader:
        print(row)
#pd.read_csv("laptop.csv")

