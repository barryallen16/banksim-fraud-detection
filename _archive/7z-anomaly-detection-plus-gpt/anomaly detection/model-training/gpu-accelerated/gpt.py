
import cudf
import cuml
import cupy as cp
import matplotlib.pyplot as plt
import seaborn as sns
from cuml.model_selection import train_test_split
from sklearn.model_selection import KFold
from cuml.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from cuml.decomposition import PCA
from cuml.cluster import DBSCAN
from cuml.neighbors import NearestNeighbors
from sklearn.metrics import (confusion_matrix, precision_score, 
                           recall_score, f1_score, accuracy_score,
                           roc_auc_score)
from sklearn.model_selection import ParameterGrid
import numpy as np
import warnings
import kagglehub
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    path = kagglehub.dataset_download("ealaxi/banksim1")
    data = cudf.read_csv(path + '/bs140513_032310.csv')
    
    data = data.drop(['zipcodeOri', 'zipMerchant'], axis=1)
    for col in ['customer', 'age', 'gender', 'merchant']:
        data[col] = data[col].str[1:-1]
    data['category'] = data['category'].str.replace('es_', '')
    data = data[data['gender'] != 'U']
    data['age'] = data['age'].replace('U', '7').astype('float32')
    data['amount'] = data['amount'].astype('float32')
    
    bins = [0, 500, 1000, 1500, 2000, 2500, 3000, float('inf')]
    labels = ['0-500', '500-1000', '1000-1500', '1500-2000', 
              '2000-2500', '2500-3000', '>3000']
    data['amount_thresh'] = cudf.cut(data['amount'], bins=bins, labels=labels)
    
    for entity in ['customer', 'merchant', 'category', 'amount_thresh']:
        freq = data.groupby(entity).size().reset_index(name=f'{entity}_freq')
        freq[f'{entity}_ID'] = cp.arange(1, len(freq) + 1)
        data = data.merge(freq, on=entity)
        if entity in ['customer', 'merchant']:
            data = data.drop(entity, axis=1)
        else:
            data[entity] = data[f'{entity}_ID']
            data = data.drop(f'{entity}_ID', axis=1)
    
    total = len(data)
    for col in data.columns:
        if '_freq' in col:
            data[col] = (data[col] / total * 100).round(5)
    
    ohe = OneHotEncoder(sparse=True, handle_unknown='ignore')
    ohe_cols = ['age', 'gender', 'category', 'amount_thresh']
    ohe_data = ohe.fit_transform(data[ohe_cols])
    ohe_data = cudf.DataFrame(ohe_data.toarray(), 
                             columns=ohe.get_feature_names(ohe_cols))
    data = cudf.concat([data.drop(ohe_cols, axis=1), ohe_data], axis=1)
    
    le = LabelEncoder()
    data['fraud'] = le.fit_transform(data['fraud'])
    
    return data
