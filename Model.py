import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import OneSidedSelection, InstanceHardnessThreshold, NearMiss, ClusterCentroids
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from keras.utils import to_categorical
from tensorflow.python.framework.random_seed import set_random_seed
from keras.callbacks import EarlyStopping
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
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


    print("Computing best subset selection...")
    sfm = SelectFromModel(LogisticRegression())
    sfm.fit(X, y)
    X = sfm.transform(X)

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    labels = set(y)
    num_classes = len(labels)
    features = X.shape[1]
    y_cat = to_categorical(y, num_classes)

    print("Undersampling data...")
    undersample = InstanceHardnessThreshold(estimator=AdaBoostClassifier(),sampling_strategy={0:50000,5:50000})
    X,y = undersample.fit_resample(X, y)
    print("Oversampling data...")
    oversample = SMOTE(sampling_strategy={1:50000,3:50000,2:50000,4:50000})
    X, y = oversample.fit_resample(X, y)


    def NN_Builder(n_layers, neurons):

        np.random.seed(123)
        set_random_seed(2)

        model = Sequential()
        model.add(Input(features))

        for i in range(n_layers):
            model.add(Dense(units=neurons, activation='relu', kernel_initializer="glorot_normal"))

        model.add(Dense(units=num_classes, activation='softmax', kernel_initializer="glorot_normal"))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['AUC'])

        return model


    es = EarlyStopping(monitor='val_loss',
                       mode='min',
                       patience=5,
                       min_delta=0.0001
                       )

    layers = [2, 3, 4]
    neurons = [60, 90, 120]
    hyperparameter_score_list = {}
    histories = {}

    print("Determining the best model...")


    for l in layers:
        for n in neurons:
            estimator = NN_Builder(l, n)
            history = estimator.fit(X, y_cat, epochs=500, batch_size=16000, validation_split=0.1, callbacks=[es])
            histories[(l, n)] = history
            hyperparameter_score_list[(l, n)] = history.history["val_auc"][-1]

    result = list(hyperparameter_score_list.items())
    result.sort(key=lambda item: item[1], reverse=True)
    bestresult = result[0][0]
    print(f"The best parameter configuration for Neural Network is: {bestresult[0]} (hidden) layers, {bestresult[1]} neurons")


    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.set_size_inches(8, 8)
    ax1.plot(histories[bestresult].history['auc'])
    ax1.plot(histories[bestresult].history['val_auc'])
    ax1.set_title('model auc')
    ax1.set_ylabel('auc')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'val'])
    ax2.plot(histories[bestresult].history['loss'])
    ax2.plot(histories[bestresult].history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'val'])
    plt.tight_layout()
    plt.show()


    print("Computing 10 fold cross validation...")

    es = EarlyStopping(
                       monitor='loss',
                       mode='min',
                       patience=5,
                       min_delta=0.0001
                       )

    performance = []
    kf = StratifiedKFold(n_splits = 10)
    for train_index, test_index in kf.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_cat_train, y_cat_test = y_cat[train_index], y_cat[test_index]
        bestmodel = NN_Builder(bestresult[0], bestresult[1])
        bestmodel.fit(X_train, y_cat_train, epochs=500, verbose=0,batch_size=16000, callbacks=[es])
        performance.append(bestmodel.evaluate(X_test, y_cat_test)[1])

    bestperformance = np.mean(performance)
    print(f"The best score is equal to = {bestperformance}")


    print("Computing precision-recall histogram...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1,random_state=42)

    y_train_cat = to_categorical(y_train, num_classes)

    bestmodel = NN_Builder(bestresult[0], bestresult[1])
    bestmodel.fit(X_train, y_train_cat, epochs=500,verbose=0, batch_size=16000, callbacks=[es])
    y_pred = bestmodel.predict(X_test)
    y_pred = [np.argmax(i) for i in y_pred]

    td = {0: "HBOND", 1: "IONIC", 2: "PICATION", 3: "PIPISTACK", 4: "SSBOND", 5: "VDW"}
    y_test_1 = [td.get(item) for item in y_test]
    y_pred_1 = [td.get(item) for item in y_pred]
    precision = precision_score(y_test_1, y_pred_1, average=None)
    precision = [i * 100 for i in precision]
    recall = recall_score(y_test_1, y_pred_1, average=None)
    recall = [i * 100 for i in recall]

    target = ["HBOND", "IONIC", "PICATION", "PIPISTACK", "SSBOND", "VDW"]

    X_axis = np.arange(len(target))

    plt.bar(X_axis - 0.2, precision, 0.4, label='Precision')
    plt.bar(X_axis + 0.2, recall, 0.4, label='Recall')

    plt.xticks(X_axis, target)
    plt.xlabel("Classes")
    plt.ylabel("Performance")
    plt.title("Precision-Recall graph")
    plt.legend()
    plt.show()


    return bestmodel, sfm, scaler