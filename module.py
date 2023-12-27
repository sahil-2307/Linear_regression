# your_ml_module.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns


def linear_reg(df,target):
    
    #Data Info
    data_info = f'There are {df.shape[0]} rows and {df.shape[1]} columns in the dataset.'
    
    #Removing columns which are object
    lis = []
    for i in df.columns:
        if df[i].dtype == 'O':
            lis.append(i)
            df.drop([i],axis=1,inplace=True)
            
    if lis == []:
        remov_cols = 'There are no non-numeric columns.'
    else:
        remov_cols = lis
    
    #Missing values
    lis2= []
    if df.isnull().sum().sum()>0:
        a = dict(df.isnull().sum()[df.isnull().sum()>0])
        b = list(a.keys())
    
    
        for i in b:
            a=list(df[i].isnull().sum()[df[i].isnull().sum()/df[i].count()*100>40])
            if a !=[]:
            #if df[i].isnull().sum()>800:
                df.drop([i],axis=1,inplace=True)
                print('Dropped column is ',i)
                continue

            c = dict(df[i].value_counts())
            d = list(c)

            if len(d)>100:
                df[i].fillna(df[i].mean(),inplace=True)
            e = d[0]
            df[i].fillna(e,inplace=True)
            cols = i
            lis2.append(cols)
        if lis2 == []:
            missing = 'There are no missing values'
        else:
            missing = f'The missing columns are{lis2}'
            
            

    import numpy as np
    from scipy.stats import chi2_contingency   #chi square
    from scipy.stats import f_oneway           #annova
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot
    
    train , test = train_test_split(df,test_size=0.2)
    train_x = train.iloc[: ,0:-1]
    train_y = train.iloc[: ,-1]
    
    test_x = test.iloc[: ,0:-1]
    test_y = test.iloc[: ,-1]
    
    linear_regression = LinearRegression()
    
    linear_regression.fit(train_x,train_y)

    intercept = linear_regression.intercept_
    coef = list(linear_regression.coef_)
    
    Rsquare=linear_regression.score(train_x,train_y)
    adjusted_r2 =  1-(1-Rsquare)*(train_x.shape[0]-1)/(train_x.shape[0]-train_x.shape[1]-1)
    
    predict_train=linear_regression.predict(train_x)
    error_train = train_y - predict_train
    mean_error_train = np.mean(error_train)
    
    MSE = np.mean(np.square(error_train))
    RMSE = np.sqrt(MSE)
    MAPE = np.mean(np.abs(error_train/train_y*100))
    
        
    predict_test = linear_regression.predict(test_x)
    error_test = test_y - predict_test
    mean_error_test = np.mean(error_test)
        
    MSE_t = np.mean(np.square(error_test))
    RMSE_t = np.sqrt(MSE_t)
    MAPE_t = np.mean(np.abs(error_test/test_y*100))
    
    
    

    
    if lis2 ==[]:
        
        return {'Dataset shape':data_info,
                'Cols':remov_cols,
                'MSE':MSE,
                'MSE':MSE,
                'RMSE':RMSE,
                'MAPE':MAPE,
                'Eqn B0': intercept,
                'Eqn Bi': coef}
    else:
        return {'Dataset shape':data_info,
                'Cols':remov_cols,
                'MSE':MSE,
                'MSE':MSE,
                'RMSE':RMSE,
                'MAPE':MAPE}
    
    
    