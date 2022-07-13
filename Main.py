import numpy as np
import pandas as pd
import os
from Bio.PDB import PDBList
from Model import model

if __name__ == '__main__':

    model, sfm = model()

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

    X = X.rank(pct=True).round(1)

    X = np.array(X)

    X = sfm.transform(X)

    y_pred = model.predict(X)
    y_pred = [np.round(i, 3) for i in y_pred]

    X = pd.DataFrame(X)
    y_pred = pd.DataFrame(y_pred)
    y_pred = y_pred.rename(
        columns={0: "HBOND", 1: "IONIC", 2: "PICATION", 3: "PIPISTACK", 4: "SSBOND", 5: "VDW"})
    data_final = pd.concat([X, y_pred], axis=1)
    print(data_final)
    data_final.to_csv(f"{pdb_id} predictions.csv")

