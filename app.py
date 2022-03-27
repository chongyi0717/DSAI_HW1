import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import RobustScaler
register_matplotlib_converters()
RANDOM_SEED = 42
test_size = 15
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
X_test=[]
y_test=[]
y_train=[]
cnt_transformer=0
def Train(csv):
    global X_test
    global cnt_transformer
    global y_train
    global y_test
    df = pd.read_csv(csv,parse_dates=['Date'], 
  index_col="Date")
    df.shape

    df['hour'] = df.index.hour
    df['day_of_month'] = df.index.day
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    df['used_power'].apply(np.ceil)
    df['produced_power'].apply(np.ceil)
    train_size = int(len(df) -16)
    
    train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]

    

    f_columns = ['淨尖峰供電能力(MW)', '尖峰負載(MW)', 'used_power','produced_power']

    f_transformer = RobustScaler()
    cnt_transformer = RobustScaler()

    f_transformer = f_transformer.fit(train[f_columns].to_numpy())
    cnt_transformer = cnt_transformer.fit(train[['operating_reserve']])

    train.loc[:, f_columns] = f_transformer.transform(train[f_columns].to_numpy())
    train['operating_reserve'] = cnt_transformer.transform(train[['operating_reserve']])

    test.loc[:, f_columns] = f_transformer.transform(test[f_columns].to_numpy())
    test['operating_reserve'] = cnt_transformer.transform(test[['operating_reserve']])

    def create_dataset(X, y, time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X.iloc[i:(i + time_steps)].values
            Xs.append(v)        
            ys.append(y.iloc[i + time_steps])
        return np.array(Xs), np.array(ys)
    time_steps = 1

    # reshape to [samples, time_steps, n_features]

    X_train, y_train = create_dataset(train, train.operating_reserve, time_steps)
    X_test, y_test = create_dataset(test, test.operating_reserve, time_steps)
    print(X_train.shape, y_train.shape)

    model = keras.Sequential()
    model.add(
    keras.layers.Bidirectional(
        keras.layers.LSTM(
        units=256, 
        input_shape=(X_train.shape[1], X_train.shape[2])
        )
    )
    )
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(
        X_train, y_train, 
        epochs=1000, 
        batch_size=128, 
        validation_split=0.1,
        shuffle=False
    )
    return model

def Test(model,csv):
    global X_test
    global cnt_transformer
    global y_train
    global y_test
    print(X_test)
    y_pred = model.predict(X_test)
    y_train_inv = cnt_transformer.inverse_transform(y_train.reshape(1, -1))
    y_test_inv = cnt_transformer.inverse_transform(y_test.reshape(1, -1))
    y_pred_inv = cnt_transformer.inverse_transform(y_pred)
    date=[20220330,20220331]
    y_p=[]
    for i in y_pred_inv:
        for j in i:
            y_p.append(int(j))
    for i in range(13):
        date.append(20220401+i)
    dict={'date':date,'operating_reserve(MW)':y_p}
    pd.DataFrame(dict).to_csv(csv,index=False)
    plt.plot(np.arange(0, len(y_train)), y_train_inv.flatten(), 'g', label="history")
    plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test_inv.flatten(), marker='.', label="true")
    plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_pred_inv.flatten(), 'r', label="prediction")
    plt.ylabel('Count')
    plt.xlabel('Time Step')
    plt.legend()
    plt.show()

# You can write code above the if-main block.
if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()

    # The following part is an example.
    # You can modify it at will.
    
    model=Train(args.training)
    Test(model,args.output)

    

