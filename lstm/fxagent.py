from ta import *

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import linear_model

class fxdata_agent:

	def read_data(path):
		df = pd.read_csv(path, sep=',', header=[0], dtype={'Local time':str, 'Open':np.float32, 'High':np.float32, 'Low':np.float32, 'Close':np.float32, 'Volume':np.float32})
		return df.drop(['Local time'], axis=1)
	
	def add_all_features(df):
		df = utils.dropna(df)
		return add_all_ta_features(df, 'Open', 'High', 'Low', 'Close', 'Volume', fillna=True)
	
	def train_test_split(df, time_steps, split_size = 0.8):
		dataset = df.values
		dataset = dataset.astype('float32')
		train_size = int(len(dataset) * 0.8) - (int(len(dataset) * 0.8) % time_steps)
		test_end_skip = (len(dataset) - train_size)	% time_steps
		return dataset[0:train_size], dataset[train_size:len(dataset)-test_end_skip]
	
	def create_lstm_trend_dataset(dataset, look_back=1, stateful=False, look_farword_ = 10, pip_decimal_place=3, debug_regression = False):
		dataX, dataY = [], []
		step=1
		regr = linear_model.LinearRegression()
		if stateful:
			step = look_back
		for i in range(0, len(dataset)-look_back-2 - look_farword_, step):
			a = dataset[i:(i+look_back)]
			dataX.append(a)
			x_vals = np.arange(look_farword_+1).reshape(-1, 1)
			y_vals = (dataset[i + look_back -1 : i + look_back + look_farword_, 3] - dataset[i + look_back -1,3] ) * math.pow(10, pip_decimal_place)
			regr.fit(x_vals, y_vals)
			dataY.append(math.atan(regr.coef_[0]) * 180 / math.pi / 9)
			if debug_regression:
				fxdata_agent.plot_regression(x_vals, y_vals, regr, 'r-angle={}'.format(math.atan(regr.coef_[0]) * 180 / math.pi / 9))
		return np.array(dataX), np.array(dataY)
	
	def create_lstm_next_close_dataset(dataset, look_back=1, stateful=False):
		dataX, dataY = [], []
		step=1
		if stateful:
			step = look_back
		for i in range(0, len(dataset)-look_back-2, step):
			a = dataset[i:(i+look_back)]
			dataX.append(a)
			dataY.append(dataset[i + look_back, 3])
		return np.array(dataX), np.array(dataY)
	
	def plot_regression(x_vals, y_vals, reg, title):
		plt.clf()
		plt.scatter(x_vals, y_vals,color='g')
		plt.title(title)
		plt.plot(x_vals, reg.predict(x_vals), color='k')
		plt.show()
	
	def plot_history(history):
		plt.clf()
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'val'], loc='upper left')
		plt.show()
		
	def plot_stateful_history(x_vals, train_loss, val_loss):
		plt.clf()
		plt.plot(x_vals, train_loss)
		plt.plot(x_vals, val_loss)
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'val'], loc='upper left')
		plt.show()
	
	