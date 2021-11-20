from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from _datetime import datetime


def svc_impl(np, XL, YL, XT, YT):
    # Select the best hyperparameters
    grid = {
        'C': np.logspace(-6, 3, 30),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': np.logspace(-6, 3, 30)
    }

    CV = GridSearchCV(estimator=SVC(),
                      param_grid=grid,
                      scoring='accuracy',
                      cv=10,
                      verbose=0)

    print(datetime.now(), " - Start search best hyperparameters")
    H = CV.fit(XL, YL)
    print(datetime.now(), " - End search best hyperparameters")

    # Learn the model with the best hyperparameters
    print(datetime.now(), " - C:", H.best_params_['C'])
    print(datetime.now(), " - kernel:", H.best_params_['kernel'])
    print(datetime.now(), " - gamma:", H.best_params_['gamma'])
    ALG = SVC(C=H.best_params_['C'],
              kernel=H.best_params_['kernel'],
              gamma=H.best_params_['gamma'])
    M = ALG.fit(XL, YL)

    # estimate the model
    YP = M.predict(XT)

    # Compute Error
    algScoreL = ALG.score(XL, YL)
    algScoreT = ALG.score(XT, YT)
    print(datetime.now(), " - ALG score learn: ", algScoreL)
    print(datetime.now(), " - ALG score test: ", algScoreT)
    err = np.mean(np.abs(YT != YP))
    print(datetime.now(), " - Model Error:", err)

    # Confusion matrix
    confusion_matrix(YT, YP)


def dt_impl(X_train, y_train, X_test, y_test):
    # Select the best hyperparameters

    grid = {
        "min_samples_split": range(2, 100),
        "min_samples_leaf": range(2, 100)
    }

    CV = GridSearchCV(estimator=DecisionTreeClassifier(),
                      param_grid=grid,
                      scoring='accuracy',
                      cv=5,
                      verbose=0)

    H = CV.fit(X_train, y_train)

    # Learn the model with the best hyperparameters
    print(datetime.now(), " - Best hyperparameters:", H.best_params_)
    ALG = DecisionTreeClassifier(
        min_samples_leaf=H.best_params_['min_samples_leaf'],
        min_samples_split=H.best_params_['min_samples_split']
    )

    M = ALG.fit(X_train, y_train)
    score_train = ALG.score(X_train, y_train)
    print(datetime.now(), " - Score Train: ", score_train)

    # estimate the model
    y_predicted = M.predict(X_test)

    # Confusion matrix
    score_test = ALG.score(X_test, y_test)
    print(datetime.now(), " - Score Test: ", score_test)
    print(datetime.now(), " - Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, y_predicted)

    title = "Score Train: " + str(score_train) + \
            "\nScore Test: " + str(score_test) + \
            "\nBest hyperparameters: " + str(H.best_params_)
    print_confusion_matrix(conf_matrix, title)

    print(datetime.now(), " - Plot Tree")
    plt.figure()
    plot_tree(M, fontsize=4)
    plt.show()


def rndm_forest_impl(X_train, y_train, X_test, y_test):
    # Select the best hyperparameters

    grid = {
        "min_samples_leaf": range(50, 100, 2),

    }

    CV = GridSearchCV(estimator=RandomForestClassifier(),
                      param_grid=grid,
                      scoring='accuracy',
                      cv=10,
                      verbose=0)

    H = CV.fit(X_train, y_train)

    # Learn the model with the best hyperparameters

    print(datetime.now(), " - min_samples_leaf:", H.best_params_['min_samples_leaf'])

    ALG = RandomForestClassifier(n_estimators=1000,
                                 min_samples_leaf=H.best_params_['min_samples_leaf'],

                                 )

    M = ALG.fit(X_train, y_train)
    print(ALG.score(X_train, y_train))
    # estimate the model

    y_predicted = M.predict(X_test)

    # Confusion matrix
    print(ALG.score(X_test, y_test))
    print(confusion_matrix(y_test, y_predicted))


def print_confusion_matrix(conf_matrix, title):
    # Print the confusion matrix using Matplotlib
    #
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.title(title)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.show()
