# COMP9417-Project
Kaggle Project Link: https://www.kaggle.com/competitions/amex-default-prediction/overview

Data set file: 
- Create a directory in the root called "data" and move all your downloaded data to that folder
- gitignore file will not track and push this folder onto the repository

Team:
- Aryan Koirala
- Vincent Wong
- Jihao (Sam) Yu

# Setting up and running the code:
Move all the files in the google drive link below and move it into the "data" folder, 
which should be empty currently.
https://drive.google.com/drive/folders/1_Cy5R8H8EtMz9uz_UIIckyDkaCvPYlTD?usp=sharing

# Project Code Information:
Note that all our code is located in the directory "project_folder." 

File Information:
Inside our project_code foler we have the following files:

cleaning_healper.py:
- Contains function that helps us manipulate the dataframe to ease our visualisation
  and cleaning of the dataset.
- Have sub-function to select and filter different columns and rows of a dataset/ returning
  a new dataset.

data_analysis.py:
- Looks at the relationship between different columns and observations in our data. For example
  using correlation to draw relations and links.

df.py:
- Helper functions to call and access different dataframes in our "data" directory.
- Also contains preprocessing data helper function.

feature_selection.py:
- Contains all the code and algorithms for our model's feature selection. Have functions
  that returns a list of features based on etc lasso, logit...

model_testing.py:
- Model testing for K-fold.

model.py:
- Contains all the code on our models as well as writing our modelling results to a csv file.
- NOTE: uncomment different lines from 219-228 to run various different models.