import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB, ComplementNB, BernoulliNB

if __name__ == '__main__':

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
    X = df[['s_rsa', 's_up', 's_down', 's_phi', 's_psi', 's_a1', 's_a2', 's_a3', 's_a4', 's_a5',
            't_rsa', 't_up', 't_down', 't_phi', 't_psi', 't_a1', 't_a2', 't_a3', 't_a4', 't_a5']]

    # Fill missing values with the most common value for that feature
    X = X.fillna({'s_rsa': X.s_rsa.mode()[0], 's_up': X.s_up.mode()[0], 's_down': X.s_down.mode()[0],
                  's_phi': X.s_phi.mode()[0], 's_psi': X.s_psi.mode()[0], 's_a1': X.s_a1.mode()[0],
                  's_a2': X.s_a2.mode()[0], 's_a3': X.s_a3.mode()[0], 's_a4': X.s_a4.mode()[0],
                  's_a5': X.s_a5.mode()[0], 't_rsa': X.t_rsa.mode()[0], 't_up': X.t_up.mode()[0],
                  't_down': X.t_down.mode()[0], 't_phi': X.t_phi.mode()[0], 't_psi': X.t_psi.mode()[0],
                  't_a1': X.t_a1.mode()[0], 't_a2': X.t_a2.mode()[0], 't_a3': X.t_a3.mode()[0],
                  't_a4': X.t_a4.mode()[0], 't_a5': X.t_a5.mode()[0]})

    # Calculate percentiles and tranform into categories
    X = X.rank(pct=True).round(1)

    # Split the dataset to define training and testing examples
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    import tensorflow
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.utils import to_categorical
    from keras.utils.vis_utils import plot_model
    from tensorflow.python.framework.random_seed import set_random_seed
    from keras.wrappers.scikit_learn import KerasClassifier
    from keras.callbacks import EarlyStopping

    labels = set(y)

    features = X.shape[1]
    num_classes = len(labels)

    y_cat = to_categorical(y, num_classes)


    def NN_Builder(n_layers, neurons):
        np.random.seed(123)
        set_random_seed(2)

        model = Sequential()
        model.add(Input(features))

        for index, i in enumerate(range(n_layers)):
            model.add(Dense(units=neurons - 5 * index, activation='relu'))

        model.add(Dense(units=num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model


    es = EarlyStopping(monitor='val_loss',
                       mode='min',
                       patience=5,
                       min_delta=0.0001
                       )

    layers = [1, 2]
    neurons = [13, 15]
    parameters = zip(layers, neurons)
    result = {}
    models = {}
    parameters_list = {}

    for i in parameters:
        model = NN_Builder(i[0], i[1])
        history = model.fit(X, y_cat, epochs=100, batch_size=16000, verbose=1, validation_split=0.2, callbacks=[es])
        models[f"{i[0]} (hidden) layers,{i[1]} neurons in the first layer"] = model
        parameters_list[f"{i[0]} (hidden) layers,{i[1]} neurons in the first layer"] = i
        result[f"{i[0]} (hidden) layers,{i[1]} neurons in the first layer"] = history.history['val_accuracy'][-1]
    result = list(result.items())
    result.sort(key=lambda item: item[1], reverse=True)
    bestresult = result[0][0]
    print(f"The best parameter configuration for Neural Network is: {bestresult}")
    models[bestresult].summary()

    bestmodel = models[bestresult]
    from sklearn.model_selection import StratifiedKFold

    kfold = StratifiedKFold(n_splits=2)
    cvscores = []

    X = np.array(X)
    y = np.array(y)
    y = to_categorical(y, num_classes)

    for train, test in kfold.split(X, np.zeros(y.shape[0])):
        model = NN_Builder(parameters_list[bestresult][0], parameters_list[bestresult][1])

        model.fit(X[train], y[train], epochs=100, batch_size=16000, verbose=1)

        scores = model.evaluate(X[test], y[test], verbose=0)
        cvscores.append(scores[1] * 100)

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))