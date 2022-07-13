import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.initializers.initializers_v2 import GlorotNormal, RandomNormal
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split

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
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.utils import to_categorical
    from tensorflow.python.framework.random_seed import set_random_seed
    from keras.callbacks import EarlyStopping
    from tensorflow.keras import initializers
    import keras_tuner as kt

    labels = set(y)

    features = X.shape[1]
    num_classes = len(labels)

    y_cat = to_categorical(y, num_classes)


    def NN_Builder(n_layers, neurons, weight_initializer, decreasing):
        np.random.seed(123)
        set_random_seed(2)

        decreasing_index = 0

        if decreasing:
            decreasing_index = math.floor(neurons / (n_layers - 1))

        model = Sequential()
        model.add(Input(features))

        for i in range(n_layers):
            model.add(Dense(units=neurons - decreasing_index * i, activation='relu',
                            kernel_initializer=weight_initializer))

        model.add(Dense(units=num_classes, activation='softmax', kernel_initializer=weight_initializer))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model


    es = EarlyStopping(monitor='val_loss',
                       mode='min',
                       patience=5,
                       min_delta=0.0001
                       )

    layers = [3, 4, 5]
    neurons = [90,70,50,30]
    decreasing = [True, False]
    batch_sizes = [16000, 32000]
    weight_initializers = ["random_normal", "glorot_normal"]
    result = {}
    models = {}
    parameters_list = {}

    print("Determining the best model...")


    for l in layers:
        for n in neurons:
            for d in decreasing:
                for b in batch_sizes:
                    for w in weight_initializers:
                        model = NN_Builder(l,n,w,d)
                        history = model.fit(X, y_cat, epochs=500, batch_size=b, validation_split=0.1, callbacks=[es])
                        models[f"layer = {l},neurons = {n},decreasing = {d},batch_size = {b},weight_initializer = {w}"] = model
                        #parameters_list[f"layer = {l},neurons = {n},batch_size = {b},weight_initializer = {w}"] =
                        result[f"layer = {l},neurons = {n},decreasing = {d},batch_size = {b},weight_initializer = {w}"] = history.history['val_accuracy'][-1]
    result = list(result.items())
    result.sort(key=lambda item: item[1], reverse=True)
    bestresult = result[0][0]
    print(f"The best parameter configuration for Neural Network is: {bestresult}")
    models[bestresult].summary()

    print("Performing 10-Fold Cross-Validation...")

    bestmodel = models[bestresult]
    from sklearn.model_selection import StratifiedKFold

    kfold = StratifiedKFold(n_splits=2)
    cvscores = []

    X = np.array(X)
    y = np.array(y)
    y = to_categorical(y, num_classes)

    for train, test in kfold.split(X, np.zeros(y.shape[0])):

        model = bestmodel#NN_Builder(parameters_list[bestresult][0], parameters_list[bestresult][1])

        model.fit(X[train], y[train], epochs=500, batch_size=16000, verbose=0)

        scores = model.evaluate(X[test], y[test], verbose=0)
        cvscores.append(scores[1] * 100)

    print(f"Accuracy with the best model is equal to: {np.mean(cvscores)} with std {np.std(cvscores)}")

    y_pred = bestmodel.predict(X)
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