import pandas as pd
import numpy  as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import scipy 
import joblib
from netCDF4 import Dataset
import netCDF4 as nc 
from tensorflow.keras.callbacks import LearningRateScheduler, Callback
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import Input 
import gc
from sklearn.metrics import mean_squared_error

def RMSE(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

def RMSE_fn(y_true, y_pred):
    return np.sqrt(np.mean(np.power(np.array(y_true, float).reshape(-1, 1) - np.array(y_pred, float).reshape(-1, 1), 2)))

def build_model():  
    inp    = Input(shape=(12,24,72,4))  
    
    x_4    = Dense(1, activation='relu')(inp)   
    x_3    = Dense(1, activation='relu')(tf.reshape(x_4,[-1,12,24,72]))
    x_2    = Dense(1, activation='relu')(tf.reshape(x_3,[-1,12,24]))
    x_1    = Dense(1, activation='relu')(tf.reshape(x_2,[-1,12]))
     
    x = Dense(64, activation='relu')(x_1)  
    x = Dropout(0.25)(x) 
    x = Dense(32, activation='relu')(x)   
    x = Dropout(0.25)(x)  
    output = Dense(24, activation='linear')(x)   
    model  = Model(inputs=inp, outputs=output)

    adam = tf.optimizers.Adam(lr=1e-3,beta_1=0.99,beta_2 = 0.99) 
    model.compile(optimizer=adam, loss=RMSE)

    return model 

def rmse(y_true, y_preds):
    return np.sqrt(mean_squared_error(y_pred = y_preds, y_true = y_true))

def score(y_true, y_preds):
    accskill_score = 0
    rmse_scores    = 0
    a = [1.5] * 4 + [2] * 7 + [3] * 7 + [4] * 6
    y_true_mean = np.mean(y_true,axis=0) 
    y_pred_mean = np.mean(y_preds,axis=0) 
#     print(y_true_mean.shape, y_pred_mean.shape)

    for i in range(24): 
        fenzi = np.sum((y_true[:,i] -  y_true_mean[i]) *(y_preds[:,i] -  y_pred_mean[i]) ) 
        fenmu = np.sqrt(np.sum((y_true[:,i] -  y_true_mean[i])**2) * np.sum((y_preds[:,i] -  y_pred_mean[i])**2) ) 
        cor_i = fenzi / fenmu
    
        accskill_score += a[i] * np.log(i+1) * cor_i
        rmse_score   = rmse(y_true[:,i], y_preds[:,i])
#         print(cor_i,  2 / 3.0 * a[i] * np.log(i+1) * cor_i - rmse_score)
        rmse_scores += rmse_score 
    
    return 2 / 3.0 * accskill_score - rmse_scores

label_path       = './data/SODA_label.nc'
nc_label         = Dataset(label_path,'r')
tr_nc_labels     = nc_label['nino'][:]

SODA_path        = './data/SODA_train.nc'
nc_SODA          = Dataset(SODA_path,'r') 

nc_sst           = np.array(nc_SODA['sst'][:])
nc_t300          = np.array(nc_SODA['t300'][:])
nc_ua            = np.array(nc_SODA['ua'][:])
nc_va            = np.array(nc_SODA['va'][:])
### 训练特征，保证和训练集一致
tr_features = np.concatenate([nc_sst[:,:12,:,:].reshape(-1,12,24,72,1),nc_t300[:,:12,:,:].reshape(-1,12,24,72,1),\
                              nc_ua[:,:12,:,:].reshape(-1,12,24,72,1),nc_va[:,:12,:,:].reshape(-1,12,24,72,1)],axis=-1)

### 训练标签，取后24个
tr_labels = tr_nc_labels[:,12:] 

### 训练集验证集划分
tr_len     = int(tr_features.shape[0] * 0.8)
tr_fea     = tr_features[:tr_len,:].copy()
tr_label   = tr_labels[:tr_len,:].copy()
 
val_fea     = tr_features[tr_len:,:].copy()
val_label   = tr_labels[tr_len:,:].copy()

#### 构建模型
model_mlp     = build_model()
#### 模型存储的位置
model_weights = './model_baseline/model_mlp_baseline.h5'
checkpoint = ModelCheckpoint(model_weights, monitor='val_loss', verbose=0, save_best_only=True, mode='min',
                             save_weights_only=True)

plateau        = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_delta=1e-4, mode='min')
early_stopping = EarlyStopping(monitor="val_loss", patience=20)
history        = model_mlp.fit(tr_fea, tr_label,
                    validation_data=(val_fea, val_label),
                    batch_size=4096, epochs=200,
                    callbacks=[plateau, checkpoint, early_stopping],
                    verbose=2)

### 模型预测
prediction = model_mlp.predict(val_fea)

print('score', score(y_true = val_label, y_preds = prediction))

