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
state_h_fwd_in = Input(batch_shape=(1, 1))
state_h_bwd_in = Input(batch_shape=(1, 1))
state_c_fwd_in = Input(batch_shape=(1, 1))
state_c_bwd_in = Input(batch_shape=(1, 1))

four_state_shape = [state_h_fwd_in, state_c_fwd_in, state_h_bwd_in, state_c_bwd_in]
two_state_shape = [state_h_fwd_in, state_h_bwd_in]

if REC == LSTM:
    rnn_out = Bidirectional( REC(1, activation='linear', use_bias=False, return_sequences=True, return_state=False, stateful=False))(features_in, initial_state=four_state_shape)
    stateful_rnn_out = Bidirectional( REC(1, activation='linear', use_bias=False, return_sequences=True, return_state=False, stateful=True))(features_in, initial_state=four_state_shape)
    rnn_inputs = [features_in, state_h_fwd_in, state_c_fwd_in, state_h_bwd_in, state_c_bwd_in]
else:
    if REC == SimpleRNN:
        rnn_out = Bidirectional( REC(1, activation='linear', use_bias=False, return_sequences=True, return_state=False, stateful=False))(features_in, initial_state=two_state_shape)
        stateful_rnn_out = Bidirectional( REC(1, activation='linear', use_bias=False, return_sequences=True, return_state=False, stateful=True))(features_in, initial_state=two_state_shape)
    else:
        rnn_out = Bidirectional( REC(1, activation='linear', use_bias=False, return_sequences=True, return_state=False, stateful=False))(features_in, initial_state=two_state_shape)
        stateful_rnn_out = Bidirectional( REC(1, activation='linear', use_bias=False, return_sequences=True, return_state=False, stateful=True))(features_in, initial_state=two_state_shape)
    rnn_inputs = [features_in, state_h_fwd_in, state_h_bwd_in]

stateless_model = Model(inputs=rnn_inputs, outputs=rnn_out)
stateful_model = Model(inputs=rnn_inputs, outputs=stateful_rnn_out)


# toy_weights = [np.asarray([[ 1.0]], dtype=np.float32), np.asarray([[0.5 ]], dtype=np.float32), np.asarray([[ -1.0 ]], dtype=np.float32), np.asarray([[ -0.5 ]], dtype=np.float32)]
# stateless_model.set_weights(toy_weights)
# stateful_model.set_weights(toy_weights)

stateful_model.set_weights( stateless_model.get_weights() )

stateful_model.save('temp_stateful.h5')
stateless_model.save('temp_stateless.h5')

x_in = np.random.normal(0,10,sequence_length)
x_in = np.asarray([1,0,0])
x_in = x_in.reshape( (1, sequence_length, feature_dim) )

fwd_initial_h = np.asarray(2.75).reshape(1,1)
fwd_initial_c = np.asarray(1.3).reshape(1,1)
bwd_initial_h = np.asarray(-2.0).reshape(1,1)
bwd_initial_c = np.asarray(-1.2).reshape(1,1)

# fwd_initial_h = np.asarray(np.random.normal(0,10)).reshape(1,1)
# fwd_initial_h = np.asarray(np.random.normal(0,10)).reshape(1,1)
# bwd_initial_h = np.asarray(np.random.normal(0,10)).reshape(1,1)
# fwd_initial_c = np.asarray(np.random.normal(0,10)).reshape(1,1)
# bwd_initial_c = np.asarray(np.random.normal(0,10)).reshape(1,1)

if REC == LSTM:
    rnn_input = [x_in, fwd_initial_h, fwd_initial_c, bwd_initial_h, bwd_initial_c]
else:
    rnn_input = [x_in, fwd_initial_h, bwd_initial_h] 
    

def print_bidi_out(non_stateful_out, stateful_out):
	fb = ['FWD::', 'BWD::']

	for i in range(2):
		print(fb[i])
		print(f'non_stateful: {non_stateful_out.T[i]}')
		print(f'stateful: {stateful_out.T[i]}')
		print(f'delta: {stateful_out.T[i]-non_stateful_out.T[i]}')

non_stateful_out = stateless_model.predict(rnn_input).reshape((sequence_length,2))
stateful_out = stateful_model.predict(rnn_input).reshape((sequence_length,2))
print_bidi_out(non_stateful_out, stateful_out)

non_stateful_out = stateless_model.predict(rnn_input).reshape((sequence_length,2))
stateful_out = stateful_model.predict(rnn_input).reshape((sequence_length,2))
print_bidi_out(non_stateful_out, stateful_out)

print('\n** RESETING STATES in STATEFUL MODEL **\n')
stateful_model.reset_states()
non_stateful_out = stateless_model.predict(rnn_input).reshape((sequence_length,2))
stateful_out = stateful_model.predict(rnn_input).reshape((sequence_length,2))
print_bidi_out(non_stateful_out, stateful_out)
