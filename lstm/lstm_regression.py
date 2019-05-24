from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model
from fxagent import fxdata_agent as fxa

import pandas as pd
import numpy as np
import pathlib

import datetime
import tensorflow as tf


print('tensorflow version: {}'.format(tf.__version__))

###########################################################################
# 						Defining Hyper-Parameters						  #
###########################################################################

print('\nDefining Hyper-Parameters...')
print('---------------------------------------------------------------') 
data_path = 'data/USDJPY_Candlestick_15_M_BID_05.05.2003-31.12.2018.csv'
checkpoint_weights_h5_path = 'checkpoints/LSTM_00000014_20190429_202523.h5'
start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
resume_from_checkpoint = False
checkpoint_autosave_period = 1
time_steps = 100
look_farword = 50
num_of_classes = 1
num_of_epochs = 100
stateful = True
add_taforex_features = True
debug_future_regression_angle = False
model_batch_size = 100
pip_decimal_place = 3
LSTM_output_units = 32
num_of_hidden_layers = 0
drop_out_value = 0.3
available_labels = ['next_close_price','future_regression_angle']
label_type = 0 # if zero = 'next_close_price' else if one = 'future_regression_angle'

#this would be redefined later at runtime
number_of_features = -1
if stateful:
	model_batch_size = 1

print('start time: {}\ndata path: {}\ncheckpoint path: {}\nresuming from checkpoint: {}\nautomatic checkpoint period: {}\npip decimal place: {}\nadd taforex features: {}\ntime steps: {}\nlook farword: {}\nnumber of classes: {}\nnumber of epochs: {}\nbatch size: {}\nLSTM stateful LSTM: {}\nhidden layers: {}\ndrop out: {}\nLSTM output units: {}\nlabel type: {}\ndebug future regression angle: {}'.format(
	datetime.datetime.now(), data_path, checkpoint_weights_h5_path, resume_from_checkpoint, checkpoint_autosave_period, pip_decimal_place, add_taforex_features, time_steps, look_farword, num_of_classes, num_of_epochs, model_batch_size, stateful, num_of_hidden_layers, drop_out_value, LSTM_output_units, available_labels[label_type], debug_future_regression_angle))


###########################################################################
# 								Reading Data 							  #
###########################################################################

print('\nReading Data...')
print('---------------------------------------------------------------') 
df = fxa.read_data(data_path)
print(df[:10])

###########################################################################
# 								Adding Features							  #
###########################################################################

if add_taforex_features:
	print('\nAdding Features...')
	print('---------------------------------------------------------------')
	df = fxa.add_all_features(df)
	print('\ninput features:')
	print('---------------------------------------------------------------')
	print(list(df.columns.values))
	print('\n{}'.format(df.head()))

	
###########################################################################
# 							Partitioning dataset 						  #
###########################################################################

print('\nPartitioning dataset...')
print('---------------------------------------------------------------') 
train_dataset, test_dataset = fxa.train_test_split(df, time_steps, 0.8)
print('Train Partition:')
print(train_dataset.shape)
print(train_dataset[:10])
print('Test Partition:')
print(test_dataset.shape)
print(test_dataset[:10])

###########################################################################
# 							Reshaping dataset 							  #
###########################################################################

print('\nReshaping dataset & creating labels...')
print('---------------------------------------------------------------') 
	
# convert an array of values into a dataset matrix
if label_type == 0:
	train_dataset, train_labels = fxa.create_lstm_next_close_dataset(train_dataset, time_steps, stateful)
	test_dataset, test_labels = fxa.create_lstm_next_close_dataset(test_dataset, time_steps, stateful)
else:
	train_dataset, train_labels = fxa.create_lstm_trend_dataset(train_dataset, time_steps, stateful, look_farword, pip_decimal_place, debug_future_regression_angle)
	test_dataset, test_labels = fxa.create_lstm_trend_dataset(test_dataset, time_steps, stateful, look_farword, pip_decimal_place, debug_future_regression_angle)

number_of_features = train_dataset.shape[2]

print('Train Partition Reshaped (Top 2):')
print(train_dataset.shape)
print(train_dataset[:2])
print('Test Partition Reshaped (Top 2):')
print(test_dataset.shape)
print(test_dataset[:2])

print('Train Partition Labels Reshaped (Top 10):')
print(train_labels.shape)
print(train_labels[:10])
print('Test Partition Labels Reshaped (Top 10):')
print(test_labels.shape)
print(test_labels[:10])

###########################################################################
# 							Building The Model 						 	  #
###########################################################################

print('\nBuilding The Model...')
print('---------------------------------------------------------------') 
def build_model():
	model = keras.Sequential()
	
	if stateful:
		model.add(layers.LSTM(units=LSTM_output_units, batch_input_shape=(model_batch_size,time_steps,number_of_features), stateful=True, return_sequences=False))
		model.add(layers.Dropout(drop_out_value))
		for i in range(0, num_of_hidden_layers):
			model.add(layers.LSTM(units=LSTM_output_units, stateful=True, return_sequences=False))
			model.add(layers.Dropout(drop_out_value))
	else:
		model.add(layers.LSTM(units=LSTM_output_units, input_shape=(time_steps,number_of_features), return_sequences=False))
		model.add(layers.Dropout(drop_out_value))
		for i in range(0, num_of_hidden_layers):
			model.add(layers.LSTM(units=LSTM_output_units, return_sequences=False))
			model.add(layers.Dropout(drop_out_value))
	
	model.add(layers.Dense(num_of_classes))
	
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model


if resume_from_checkpoint:
	print('Resuming from checkpoint: {}'.format(checkpoint_weights_h5_path))
	model = load_model(checkpoint_weights_h5_path)
else:
	model = build_model()

print(model.inputs)
model.summary()

###########################################################################
# 						Testing 10 widthed batch... 					  #
###########################################################################

print('\nTesting 10 widthed batch...')
print('---------------------------------------------------------------')
example_batch = train_dataset[:10]
print('Real Labels:')
print(train_labels[:10])
example_result = model.predict(example_batch, batch_size=model_batch_size, verbose=0)
print('Predicted Labels:')
print(example_result)
	
###########################################################################
# 							Fitting The Model 							  #
###########################################################################

print('\nFitting the model...')
print('---------------------------------------------------------------')

if stateful:
	epochs = np.arange(num_of_epochs)
	train_loss = np.zeros(num_of_epochs)
	val_loss = np.zeros(num_of_epochs)
	
	for i in range(1, num_of_epochs+1):
		print('stateful training epoch {}/{}'.format(i,num_of_epochs))
		history = model.fit(train_dataset, train_labels, epochs=1, validation_split = 0.2, verbose=1, batch_size=model_batch_size, shuffle=False)
		train_loss[i-1] = history.history['loss'][0]
		val_loss[i-1] = history.history['val_loss'][0]
		model.reset_states()
		if i%checkpoint_autosave_period ==0:
			model.save('checkpoints/LSTM_stateful{:08d}_{}.h5'.format(i, start_time))
			
	fxa.plot_stateful_history(epochs, train_loss, val_loss)
	
else:
	mc = keras.callbacks.ModelCheckpoint('checkpoints/LSTM_stateless{epoch:08d}_' + start_time + '.h5', save_weights_only=False, save_best_only=True, period=checkpoint_autosave_period)	
	history = model.fit(train_dataset, train_labels, epochs=num_of_epochs, validation_split = 0.2, verbose=1, callbacks=[mc], batch_size=model_batch_size)
	fxa.plot_history(history)
	
###########################################################################
# 							Evaluating The Model						  #
###########################################################################

print('\nEvaluating the model...')
print('---------------------------------------------------------------')
example_batch = test_dataset[:10]
print(test_labels[:10])
example_result = model.predict(example_batch, batch_size=model_batch_size, verbose=1)
print(example_result)
loss = model.evaluate(test_dataset, test_labels, batch_size=model_batch_size, verbose=0)
print('Testing set MSE: {:5.2f} '.format(loss))


print('\nSaving final weights...')
model.save('checkpoints/LSTM_Final_{}.h5'.format(start_time))
print('Finished.')

