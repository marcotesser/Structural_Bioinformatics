import numpy as np
import pandas as pd
import os
from Bio.PDB import PDBList
from Model import model

if __name__ == '__main__':

    model, sfm, scaler = model()

    pdb_id = input("Enter your protein: ")

    pdblist = PDBList()

    pdblist.retrieve_pdb_file(pdb_id, pdir=".")

    print("Protein download completed")

    os.system(f"python calc_features.py {pdb_id}.cif")

    df = pd.read_csv("./" + pdb_id + ".tsv", sep='\t')

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

    X = sfm.transform(X)

    X = scaler.transform(X)

    y_pred = model.predict(X)
    y_pred = [np.round(i, 3) for i in y_pred]

    X = pd.DataFrame(X)
    y_pred = pd.DataFrame(y_pred)

    print(y_pred)

    y_pred = y_pred.rename(
        columns={0: "HBOND", 1: "IONIC", 2: "PICATION", 3: "PIPISTACK", 4: "SSBOND", 5: "VDW"})

    support = sfm.get_support()
    features = ['s_up', 's_down', 's_phi', 's_psi', 's_a1', 's_a2', 's_a3', 's_a4', 's_a5',
                't_up', 't_down', 't_phi', 't_psi', 't_a1', 't_a2', 't_a3', 't_a4', 't_a5']

    features = [features[i] for i in range(len(features)) if support[i] == True]

    data_final = pd.concat([X, y_pred], axis=1)

    data_final = data_final.rename(
        columns={0: features[0], 1: features[1], 2: features[2], 3: features[3], 4: features[4], 5: features[5]})

    print(data_final)
    data_final.to_csv(f"{pdb_id} predictions.csv")

