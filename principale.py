import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from algorithms import svc_impl
from algorithms import dt_impl
from algorithms import rndm_forest_impl

if __name__ == "__main__":
    # -----------------------------------
    # EDA
    # -----------------------------------
    dataset = pd.read_csv("heart.csv")
    pd.set_option('display.max_columns', None)
    pd.options.display.width = None
    pd.options.mode.chained_assignment = None
    #print("Number of rows:",dataset.shape[0])
    #print("Number of columns:",dataset.shape[1])
    #print(dataset.info())
    #print(dataset.describe().T)
    #print(dataset.describe(include=object).T)

    #plt.figure(figsize=(12, 8))
    #sns.heatmap(dataset.corr(), annot=True)
    #plt.show()

    # ------------------------------------------
    # Feature Engineering
    # ------------------------------------------

    dataset["Sex"] = [1 if i == "M" else 0 for i in dataset["Sex"]]
    dataset["ExerciseAngina"] = [1 if i == "Y" else 0 for i in dataset["ExerciseAngina"]]
    for i in range(len(dataset["RestingECG"])):
        if dataset["RestingECG"][i] == "Normal":
            dataset["RestingECG"][i] = 0
        elif dataset["RestingECG"][i] == "LVH":
            dataset["RestingECG"][i] = 1
        elif dataset["RestingECG"][i] == "ST":
            dataset["RestingECG"][i] = 2

    for i in range(len(dataset["ST_Slope"])):
        if dataset["ST_Slope"][i] == "Up":
            dataset["ST_Slope"][i] = 0
        elif dataset["ST_Slope"][i] == "Flat":
            dataset["ST_Slope"][i] = 1
        elif dataset["ST_Slope"][i] == "Down":
            dataset["ST_Slope"][i] = 2


    for i in range(len(dataset["ChestPainType"])):
        if dataset["ChestPainType"][i] == "ATA":
            dataset["ChestPainType"][i] = 0
        elif dataset["ChestPainType"][i] == "NAP":
            dataset["ChestPainType"][i] = 1
        elif dataset["ChestPainType"][i] == "ASY":
            dataset["ChestPainType"][i] = 2
        elif dataset["ChestPainType"][i] == "TA":
            dataset["ChestPainType"][i] = 3


    #print(dataset.sample(10))

    # -------------------------------------
    # Train/Test division
    # -------------------------------------

    X = dataset.drop(["HeartDisease"], axis=1)
    y = dataset["HeartDisease"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    scaler = preprocessing.MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)



    print("--------------------------------SVC-------------------------------------------------------------")
    #svc_impl(np, X_train_scaled, y_train, X_test_scaled, y_test)
    #dt_impl(np, X_train_scaled, y_train, X_test_scaled, y_test)
    rndm_forest_impl(np, X_train_scaled, y_train, X_test_scaled, y_test)



