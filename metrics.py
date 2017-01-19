from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import numpy as np
import math

def continuous_metrics(true, pred, msg):
    ndigit = 3
    true, pred = true, pred
    true = np.array(true)
    pred = np.array(pred)
  

   

    MAE= mean_absolute_error(true, pred)
    Pearson_r = pearsonr(true, pred)


    print('MAE, Pearson_r')
    

    MAE=round(MAE, ndigit)
    Pearson_r=round(Pearson_r[0], ndigit)
    

    return (MAE, Pearson_r)
    
