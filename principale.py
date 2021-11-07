import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    # -----------------------------------
    # EDA
    # -----------------------------------
    dataset = pd.read_csv("heart.csv")
    pd.set_option('display.max_columns', None)
    pd.options.display.width = None
    pd.options.mode.chained_assignment = None
    print("Number of rows:",dataset.shape[0])
    print("Number of columns:",dataset.shape[1])
    print(dataset.info())
    print(dataset.describe().T)
    print(dataset.describe(include=object).T)

    plt.figure(figsize=(12, 8))
    sns.heatmap(dataset.corr(), annot=True)
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


    print(dataset.sample(10))

    # -------------------------------------
    # Train/Test division
    # -------------------------------------

    X = dataset.drop(["HeartDisease"], axis=1)
    y = dataset["HeartDisease"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

    scaler = preprocessing.MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -----------------------------
    # Chosen Algorithm: SVC
    # -----------------------------

    # Select the best hyperparameters

    grid = {'C' : np.logspace(-4, 3, 10),
            'kernel' : ['linear','rbf'],
            'gamma' : np.logspace(-4, 3, 10)}

    CV = GridSearchCV( estimator  = SVC(),
                       param_grid = grid,
                       scoring    = 'accuracy',
                       cv         = 10,
                       verbose    = 0)

    H = CV.fit(X_train,y_train)

    # Learn the model with the best hyperparameters

    ALG = SVC(C      = H.best_params_['C'],
              kernel = H.best_params_['kernel'],
              gamma  = H.best_params_['gamma'])

    M = ALG.fit(X_train, y_train)

    # estimate the model

    y_predicted = M.predict(X_test)

    # Confusion matrix

    print(confusion_matrix(y_test,y_predicted))


