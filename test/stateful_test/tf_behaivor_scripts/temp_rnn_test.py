import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=''

import numpy as np
from tensorflow.keras.layers import Input, Dense, SimpleRNN, GRU, LSTM, Bidirectional
from tensorflow.keras.models import Model

USE_TOY_WEIGHTS = True
REC_LAYER = GRU
sequence_length = 3
feature_dim = 1
features_in = Input(batch_shape=(1, sequence_length, feature_dim)) 
state_h_in = Input(batch_shape=(1, 1))

rnn_out = REC_LAYER(1, activation=None,  use_bias=False, return_sequences=True, return_state=False, stateful=False)(features_in, initial_state=state_h_in)
stateless_model = Model(inputs=[features_in, state_h_in], outputs=rnn_out)

stateful_rnn_out = REC_LAYER(1, activation=None,  use_bias=False, return_sequences=True, return_state=False, stateful=True)(features_in, initial_state=state_h_in)
stateful_model = Model(inputs=[features_in, state_h_in], outputs=stateful_rnn_out)

if USE_TOY_WEIGHTS:
	if REC_LAYER == SimpleRNN:
		toy_weights = [ np.asarray([[1.0]], dtype=np.float32), np.asarray([[-0.5]], dtype=np.float32)]

	elif REC_LAYER == GRU:
		# for a GRU, the first are the non-recurrent kernels W, and the second are the recurrent kernels U (V)
		toy_weights = [np.asarray([[ 1.0, -2.0,  3.0 ]], dtype=np.float32), np.asarray([[ -0.5 , 2.0, -1.1 ]], dtype=np.float32)]

	stateless_model.set_weights(toy_weights)
	stateful_model.set_weights(toy_weights)

# w = stateless_model.get_weights()
# print(w)

stateless_model.save('temp_stateless.h5', include_optimizer=False)
stateful_model.save('temp_stateful.h5', include_optimizer=False)

x_in = np.zeros(sequence_length)
x_in[0] = 1
x_in = x_in.reshape( (1, sequence_length, feature_dim) )
initial_state = np.asarray( [10])
initial_state = initial_state.reshape((1,1))

def print_rnn_out(non_stateful_out, stateful_out):
	fb = ['FWD::', 'BWD::']

	print(f'non_stateful: {non_stateful_out}')
	print(f'stateful: {stateful_out}')
	print(f'delta: {stateful_out-non_stateful_out}')

non_stateful_out = stateless_model.predict([x_in, initial_state]).reshape((sequence_length))
stateful_out = stateful_model.predict([x_in, initial_state]).reshape((sequence_length))
print_rnn_out(non_stateful_out, stateful_out)

non_stateful_out = stateless_model.predict([x_in, initial_state]).reshape((sequence_length))
stateful_out = stateful_model.predict([x_in, initial_state]).reshape((sequence_length))
print_rnn_out(non_stateful_out, stateful_out)

print('\n** RESETING STATES in STATEFUL MODEL **\n')
stateful_model.reset_states()
non_stateful_out = stateless_model.predict([x_in, initial_state]).reshape((sequence_length))
stateful_out = stateful_model.predict([x_in, initial_state]).reshape((sequence_length))
print_rnn_out(non_stateful_out, stateful_out)

non_stateful_out = stateless_model.predict([x_in, initial_state]).reshape((sequence_length))
stateful_out = stateful_model.predict([x_in, initial_state]).reshape((sequence_length))
print_rnn_out(non_stateful_out, stateful_out)
