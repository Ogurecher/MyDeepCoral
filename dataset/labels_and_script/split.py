import pandas as pd
import os

def createDir(name):
    if not os.path.exists(name):
        os.makedirs(name)

labels = pd.read_csv("labels.csv", delimiter=';', header=None, names=["name", "label"])

for row in labels.iterrows():
    createDir(row[1]["label"].strip())
    source = os.path.dirname(os.path.abspath(__file__)) + "\\images\\" + row[1]["name"] + ".png"
    destination = os.path.dirname(os.path.abspath(__file__)) + "\\" + row[1]["label"].strip() + "\\" + row[1]["name"] + ".png"
    if os.path.exists(source):
        os.rename(source, destination)