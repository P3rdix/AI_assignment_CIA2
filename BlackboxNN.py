import numpy as np
import tensorflow as tf
import pandas as pd

df = pd.read_csv('newWeaatherHistory.csv')
 
df.head()

train_df = df.sample(frac=0.75, random_state=4)

val_df = df.drop(train_df.index)

max_val = train_df.max(axis= 0)
min_val = train_df.min(axis= 0)
 
range = max_val - min_val
train_df = (train_df - min_val)/(range)
 
val_df =  (val_df- min_val)/range

X_train = train_df.drop('Temperature (C)',axis=1)
X_val = val_df.drop('Temperature (C)',axis=1)
y_train = train_df['Temperature (C)']
y_val = val_df['Temperature (C)']

input_shape = [X_train.shape[1]]
 
input_shape

model = tf.keras.Sequential([tf.keras.layers.Dense(units=1,input_shape=input_shape)])

model.summary()

model = tf.keras.Sequential([tf.keras.layers.Dense(units=64, activation='relu',input_shape=input_shape),tf.keras.layers.Dense(units=64, activation='relu'),tf.keras.layers.Dense(units=1)])

model.summary()

model.compile(optimizer='adam',loss='mae')

losses = model.fit(X_train, y_train,validation_data=(X_val, y_val), batch_size=256,epochs=15)

model.predict(X_val.iloc[0:3, :])

print(y_val.iloc[0:3])

loss_df = pd.DataFrame(losses.history)

loss_df.loc[:,['loss','val_loss']].plot()