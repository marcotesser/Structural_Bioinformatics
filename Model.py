import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.initializers.initializers_v2 import GlorotNormal, RandomNormal
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score


def model():


    dfs = []
    for filename in os.listdir('data/features_ring'):
        dfs.append(pd.read_csv('data/features_ring/' + filename, sep='\t'))
    df = pd.concat(dfs)

    # Dropping examples with missing label
    df = df[df.Interaction.notna()]

    # Replacing examples' labels with integer values
    di = {"HBOND": 0, "IONIC": 1, "PICATION": 2, "PIPISTACK": 3, "SSBOND": 4, "VDW": 5}
    df = df.replace({"Interaction": di})

    # Extract target feature
    y = df['Interaction']
    y

    # Extract training features
    X = df[['s_up', 's_down', 's_phi', 's_psi', 's_a1', 's_a2', 's_a3', 's_a4', 's_a5',
            't_up', 't_down', 't_phi', 't_psi', 't_a1', 't_a2', 't_a3', 't_a4', 't_a5']]

    # Fill missing values with the most common value for that feature
    X = X.fillna({'s_up': X.s_up.mode()[0], 's_down': X.s_down.mode()[0],
                  's_phi': X.s_phi.mode()[0], 's_psi': X.s_psi.mode()[0], 's_a1': X.s_a1.mode()[0],
                  's_a2': X.s_a2.mode()[0], 's_a3': X.s_a3.mode()[0], 's_a4': X.s_a4.mode()[0],
                  's_a5': X.s_a5.mode()[0], 't_up': X.t_up.mode()[0],
                  't_down': X.t_down.mode()[0], 't_phi': X.t_phi.mode()[0], 't_psi': X.t_psi.mode()[0],
                  't_a1': X.t_a1.mode()[0], 't_a2': X.t_a2.mode()[0], 't_a3': X.t_a3.mode()[0],
                  't_a4': X.t_a4.mode()[0], 't_a5': X.t_a5.mode()[0]})

    # Calculate percentiles and tranform into categories
    X = X.rank(pct=True).round(1)



    import tensorflow
    import math
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.utils import to_categorical
    from tensorflow.python.framework.random_seed import set_random_seed
    from keras.callbacks import EarlyStopping
    from tensorflow.keras import initializers
    import keras_tuner as kt



    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=123,
                                                        stratify=y)

    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import LogisticRegression
    sfm = SelectFromModel(LogisticRegression())
    sfm.fit(X, y)
    X_train = sfm.transform(X_train)
    X_test = sfm.transform(X_test)

    labels = set(y)
    features = X_train.shape[1]
    num_classes = len(labels)
    y_cat = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    def NN_Builder(n_layers=0, neurons=0):
        np.random.seed(123)
        set_random_seed(2)

        model = Sequential()
        model.add(Input(features))

        for i in range(n_layers):
            model.add(Dense(units=neurons, activation='relu', kernel_initializer="glorot_normal"))

        model.add(Dense(units=num_classes, activation='softmax', kernel_initializer="glorot_normal"))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model


    es = EarlyStopping(monitor='loss',
                       mode='min',
                       patience=5,
                       min_delta=0.0001
                       )

    layers = [2, 3]
    neurons = [90, 60, 30]
    hyperparameter_score_list = {}

    print("Determining the best model...")


    for l in layers:
        for n in neurons:

            estimator = KerasRegressor(build_fn=NN_Builder, n_layers=l, neurons=n, epochs=500, batch_size=16000, callbacks=[es])
            results = cross_val_score(estimator, X_train, y_cat,scoring="accuracy", cv=10)
            hyperparameter_score_list[(n, l)] = results.mean()

    result = list(hyperparameter_score_list.items())
    result.sort(key=lambda item: item[1], reverse=True)
    bestresult = result[0][0]
    print(f"The best parameter configuration for Neural Network is: {bestresult[0]} layers and {bestresult[1]} neurons")
    bestmodel = NN_Builder(bestresult[0], bestresult[1])
    bestmodel.fit(X_train, y_cat, epochs=500, batch_size=16000, callbacks=[es])
    result = bestmodel.evaluate(X_test, y_test)
    print(f"best accuracy with best model is equal to {result[1]}")
    y_pred = bestmodel.predict(X_test)
    y_pred = [np.argmax(i) for i in y_pred]
    target = ["HBOND", "IONIC", "PICATION", "PIPISTACK", "SSBOND", "VDW"]

    # set plot figure size
    fig, c_ax = plt.subplots(1, 1, figsize=(12, 8))
    lb = LabelBinarizer()
    lb.fit(y)
    y_test = lb.transform(y)
    y_pred = lb.transform(y_pred)

    for (idx, c_label) in enumerate(target):
        fpr, tpr, thresholds = roc_curve(y[:, idx].astype(int), y_pred[:, idx])
        c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (c_label, auc(fpr, tpr)))
    c_ax.plot(fpr, fpr, 'b-', label='Random Guessing')

    print('ROC AUC score:', roc_auc_score(y, y_pred, average="macro"))

    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    plt.show()


    return bestmodel