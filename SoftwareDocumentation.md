# Software tutorial
### Authors: Marco Tesser, Emma Roveroni

## 1 Files
- *Main.py*: the main python script that has to be run to start the execution. Contains all the code to upload and process the protein and later classify its contacts via the fully trained model.
- *Model.py*: python script that contains the untrained model. In order to train the model the Main.py script has to be executed.
- *calc-features.py*: python script to compute the proteins features.
- *data*: folder containing all of the training data in .tsv format.
- *ramachandran.dat*: utility file to compute features.
- *atchley.tsv*: utility file to compute features. 

## 2 Dependecies
The software uses different python libraries to be run, and these must be all installed beforehand in order to run the program correctly. The dependecies are the following:
- Python 3.x 
- Numpy
- Pandas
- Sci-kit Learn 
- Keras
- Biopython
- Matplotlib
- Imbalanced-learn

## 3 How to run software
*Premise*: In order to run the software a python compatible IDE is required (e.g PyCharm).

To run the software is sufficient to run the Main.py script. Firstly this will train the classification model (found in model.py) determining which are the best hyperparameters for it and then display some statistics regarding the just trained model.
Once the training has completed the software will ask the user for which protein he/she would like to predict the contacts type.
The user will have to input the 4 letter code that identifies the protein in the PDB. Once this is done the software will proceed to complete all of steps necessary to predict the proteins' contacts. Once it has finished it will output a .csv file containing the features and probability distribution for each contact of the protein.

