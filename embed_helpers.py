import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
from tensorflow import set_random_seed
set_random_seed(42)

from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout
from keras.layers.embeddings import Embedding
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.externals import joblib

import pickle


def make_embedding_network():
    
    '''
    Defines the embedding network model
    '''
        
    inputs=[]
    embeddings=[]
    
    # embeddings for work class
    wc_input = Input(shape=(1,))
    inputs.append(wc_input)
    wc_embedding = Embedding(7, 3, input_length=1, name='work_class_embedding')(wc_input)
    wc_embedding = Reshape(target_shape=(3,))(wc_embedding)
    embeddings.append(wc_embedding)
    
    # embeddings for education
    edu_input = Input(shape=(1,))
    inputs.append(edu_input)
    edu_embedding = Embedding(16, 4, input_length=1, name='education_embedding')(edu_input)
    edu_embedding = Reshape(target_shape=(4,))(edu_embedding)
    embeddings.append(edu_embedding)    
        
    # embeddings for marital status
    ms_input = Input(shape=(1,))
    inputs.append(ms_input)
    ms_embedding = Embedding(7, 3, input_length=1, name='marital_status_embedding')(ms_input)
    ms_embedding = Reshape(target_shape=(3,))(ms_embedding)
    embeddings.append(ms_embedding)
    
    # embeddings for occupation
    oc_input = Input(shape=(1,))
    inputs.append(oc_input)
    oc_embedding = Embedding(14, 3, input_length=1, name='occupation_embedding')(oc_input)
    oc_embedding = Reshape(target_shape=(3,))(oc_embedding)
    embeddings.append(oc_embedding)
    
    # embeddings for relationship
    rel_input = Input(shape=(1,))
    inputs.append(rel_input)
    rel_embedding = Embedding(6, 3, input_length=1, name='relationship_embedding')(rel_input)
    rel_embedding = Reshape(target_shape=(3,))(rel_embedding)
    embeddings.append(rel_embedding)
    
    # embeddings for race
    race_input = Input(shape=(1,))
    inputs.append(race_input)
    race_embedding = Embedding(5, 2, input_length=1, name='race_embedding')(race_input)
    race_embedding = Reshape(target_shape=(2,))(race_embedding)
    embeddings.append(race_embedding)
    
    # embedding for sex
    sex_input = Input(shape=(1,))
    inputs.append(sex_input)
    sex_embedding = Embedding(2,1, input_length=1, name='sex_embedding')(sex_input)
    sex_embedding = Reshape(target_shape=(1,))(sex_embedding)
    embeddings.append(sex_embedding)
    
    # embeddings for country
    country_input = Input(shape=(1,))
    inputs.append(country_input)
    country_embedding = Embedding(41, 10, input_length=1, name='country_embedding')(country_input)
    country_embedding = Reshape(target_shape=(10,))(country_embedding)
    embeddings.append(country_embedding)  
    
    # numeric input
    numeric_input = Input(shape=(6,))
    inputs.append(numeric_input)
    numeric_embedding = Dense(4)(numeric_input)    
    embeddings.append(numeric_embedding)
    
    # Concatenate all the embedding layers
    model = Concatenate()(embeddings)
    
    # add some hidden layers with dropout
    model = Dense(25, activation='relu')(model)
    model = Dropout(0.25)(model)
    model = Dense(15, activation='relu')(model)
    model = Dropout(.1)(model)
    model = Dense(10, activation='relu')(model)
    model = Dropout(.1)(model)
    
    # get the output through a sigmoid
    output = Dense(1, activation='sigmoid')(model)
    
    # make the final model
    final_model = Model(inputs, output)

    # set training process
    final_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    
    return final_model

def make_input_list(df, cat_names):
    
    '''
    Defines the inputs according the way the embedding network expects.
    
    Parameters:
    ----------
    df: pandas dataframe containing features as columns
    cat_names: list, contains the names of categorical columns
    
    Returns:
    -------
    A list that contains n+1 numpy arrays.    
    let M = df.shape[1], N = df.shape[0]    
    First n arrays are of the shape (M,). Each of them is a categorical column.
    Last array is of the shape (M, N-len(cat_names). This is the subarray of numeric columns.
    '''
    
    cat_inputs = [df[cat_column].values.astype(int) for cat_column in cat_names]
    numeric_inputs = [df.loc[:, list(df.columns.difference(cat_names))].values.astype(np.float64)]
    
    return cat_inputs + numeric_inputs


def pre_process_data(df_X, cat_columns, le_path='models/label_encoders.pickle'):
        
    '''
    The embedding layers expect each categorical variable to be of an integer, which can be achieved by label encoding. 
    This function uses LabelEncoder implementation in scikit-learn.
    
    Parameters:
    ----------
    
    df_x: total data set, as a pandas dataframe
    
    cat_columns: a list containing the names of categorical columns
                 in this case: cat_columns=['workclass_cat',
                                 'education_cat',
                                 'marital.status_cat',
                                 'occupation_cat',
                                 'relationship_cat',
                                 'race_cat',
                                 'sex_cat',
                                 'native.country_cat']
                                 
    le_path: path to store the list of all LabelEncoder objects
    
    Returns:
    -------
     
    df_X: A pandas dataframe that has the categorical columns label-encoded. Also saves the LabelEncoder object list to le_path.    
    '''
    les = []
    for i in cat_columns:
        le = LabelEncoder()
        le.fit(df_X.loc[:, i])
        les.append(le)
        df_X.loc[:, i] = le.transform(df_X.loc[:, i])
    
    with open(le_path, 'wb') as f:
        pickle.dump(les, f, -1)
    print("label encoders saved to: "+ le_path)
        
        
    return df_X

def normalize_numeric_columns(df, columns,scaler=StandardScaler(), is_production=False, scaler_path=None):
    
    '''
    Many models expect the input features to be normalized.
    This function normalizes the numeric column values.
    
    Parameters:
    ----------
    df: pandas dataframe
    columns: list of column names that needs to be normalized. These should be numeric columns.
    scaler: scikitlearn StandardScaler() object
    is_production: In a production setting, we might receive the examples one at a time.
                    When set to true, the StandardScaler() object that fitted on this data set will be saved to scaler_path so that it can be used later to transform single instances.
    scaler_path: path to save the StandardScaler() object
    
    Returns:
    -------
    A dataframe with all the columns as df and numeric columns normalized.
    Also saves the scaler to scaler_path.
    '''
    
    original_cols = df.columns
    df_numeric = df.loc[:,columns]
    df_cat = df.drop(columns, axis=1)
        
    if not is_production:
        scaler.fit(df_numeric)
        
    df_numeric = scaler.transform(df_numeric)
    df_val = np.concatenate([df_cat.values, df_numeric],axis=1)
        
    if scaler_path is not None:
        joblib.dump(scaler, scaler_path)
        print("scaler saved to " + scaler_path)
        
    return pd.DataFrame(df_val, columns=original_cols)
        
        
    

def save_model(model, model_path, weights_path):
    '''
    Saves the learned model.
    
    Parameters:
    ----------
    model: keras.engine.training.Model object, model to be saved
    model_path: where to save the json file of the model
    weights_path: where to save the weights as hdf5
    
    Returns:
    -------
    Saves model in json format in model_path.
    Saves model weights in hdf5 format in weights_path.
    '''
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_path, "w") as json_file:
        json_file.write(model_json)
    print("Saved model to " + model_path)
    # serialize weights to HDF5
    model.save_weights(weights_path)
    print("Saved model weights to " + weights_path)
    

def pca_and_visualize(embedding_matrix, labels):
    
    '''
    Apply PCA on an embedding matrix to reduce the dimensionality to 2 and plot them along with the respective category name.
    
    Parameters:
    ----------
    embedding_matrix: a numpy array, supposed to be the vector space learned by the entity embedding network for a category.
    labels: list of category values the particular categorical variable has had.
    
    Returns:
    -------
    None.
    Plots a visualization of the 2D repersentation of the category embeddings.    
    '''
    
    print('\n-----------------------------------\ninitial shape: '+ str(embedding_matrix.shape))
    print('Reducing the dimensionility of vectors from {0} to 2 ...\n'.format(embedding_matrix.shape[1]))
    pca = PCA(n_components=2)
    pca.fit(embedding_matrix)
    #2d transformation
    Y = pca.transform(embedding_matrix)
    
    #plotting
    plt.figure();
    plt.scatter(-Y[:, 0], -Y[:, 1])
    for i, txt in enumerate(labels):
        plt.annotate(txt, (-Y[i, 0],-Y[i, 1]), xytext = (-20, 8), textcoords = 'offset points')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.show()