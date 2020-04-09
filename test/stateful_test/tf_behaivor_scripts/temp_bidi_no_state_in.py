import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=''

import numpy as np
from tensorflow.keras.layers import Input, Dense, SimpleRNN, GRU, LSTM, Bidirectional
from tensorflow.keras.models import Model

REC = LSTM

sequence_length = 3
feature_dim = 1
features_in = Input(batch_shape=(1, sequence_length, feature_dim)) 

rnn_out = Bidirectional( REC(1, activation=None, use_bias=False, return_sequences=True, return_state=False, stateful=False))(features_in)
stateless_model = Model(inputs=[features_in], outputs=[rnn_out])

stateful_rnn_out = Bidirectional( REC(1, activation=None, use_bias=False, return_sequences=True, return_state=False, stateful=True))(features_in)
stateful_model = Model(inputs=features_in, outputs=stateful_rnn_out)

stateful_model.set_weights( stateless_model.get_weights() )

x_in = np.random.normal(0,10,sequence_length)
x_in = x_in.reshape( (1, sequence_length, feature_dim) )

def print_bidi_out(non_stateful_out, stateful_out):
	fb = ['FWD::', 'BWD::']

	for i in range(2):
		print(fb[i])
		print(f'non_stateful: {non_stateful_out.T[i]}')
		print(f'stateful: {stateful_out.T[i]}')
		print(f'delta: {stateful_out.T[i]-non_stateful_out.T[i]}')


non_stateful_out = stateless_model.predict(x_in).reshape((sequence_length,2))
stateful_out = stateful_model.predict(x_in).reshape((sequence_length,2))
print_bidi_out(non_stateful_out, stateful_out)

non_stateful_out = stateless_model.predict(x_in).reshape((sequence_length,2))
stateful_out = stateful_model.predict(x_in).reshape((sequence_length,2))
print_bidi_out(non_stateful_out, stateful_out)

print('\n** RESETING STATES in STATEFUL MODEL **\n')
stateful_model.reset_states()
non_stateful_out = stateless_model.predict(x_in).reshape((sequence_length,2))
stateful_out = stateful_model.predict(x_in).reshape((sequence_length,2))
print_bidi_out(non_stateful_out, stateful_out)
