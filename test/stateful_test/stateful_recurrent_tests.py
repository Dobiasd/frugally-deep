import h5py
import keras
import numpy as np
from keras.layers import Input, Dense, GRU, LSTM, Bidirectional
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

PRINT_SUMMARIES = False
VERBOSE = True

def get_trained_model(x_train, y_train, layer_name, n_recurrent_units, bidi):
	if layer_name == 'LSTM':
		REC_LAYER = LSTM
	else:
		REC_LAYER = GRU
	######  Define/Build/Train Training Model
	training_in_shape = x_train.shape[1:]
	training_in = Input(shape=training_in_shape)
	if bidi:
		recurrent_out = Bidirectional( REC_LAYER(n_recurrent_units, return_sequences=True, stateful=False) )(training_in)
	else:
		recurrent_out = REC_LAYER(n_recurrent_units, return_sequences=True, stateful=False)(training_in)
	training_pred = Dense(1)(recurrent_out)
	training_model = keras.Model(inputs=training_in, outputs=training_pred)
	training_model.compile(loss='mean_squared_error', optimizer='adam')
	if PRINT_SUMMARIES:
		training_model.summary()
	training_model.fit(x_train, y_train, batch_size=2, epochs=10, verbose=0)
	trained_weights = training_model.get_weights()
	return training_model, trained_weights

def get_test_model(n_recurrent_units, sequence_length, feature_dim, layer_name, stateful, initialize_states, weights, bidi):
	features_in = Input(batch_shape=(1,sequence_length,feature_dim))  ## stateful ==> needs batch_shape specified
	if layer_name == 'LSTM':
		REC_LAYER = LSTM
	else:
		REC_LAYER = GRU
	if bidi:
		if not initialize_states:
			recurrent_out = Bidirectional( REC_LAYER(n_recurrent_units, return_sequences=True, stateful=stateful ) )(features_in)
			pred = Dense(1)(recurrent_out)
			test_model = keras.Model(inputs=features_in, outputs=pred)
		else: 
			state_h_fwd_in = Input(batch_shape=(1,n_recurrent_units))
			state_h_bwd_in = Input(batch_shape=(1,n_recurrent_units))
			state_c_fwd_in = Input(batch_shape=(1,n_recurrent_units))
			state_c_bwd_in = Input(batch_shape=(1,n_recurrent_units))
			if layer_name == 'LSTM':
				recurrent_out = Bidirectional( REC_LAYER(n_recurrent_units, return_sequences=True, stateful=stateful) )(features_in, initial_state=[state_h_fwd_in, state_h_bwd_in, state_c_fwd_in, state_c_bwd_in])
				pred = Dense(1)(recurrent_out)
				test_model = keras.Model(inputs=[features_in, state_h_fwd_in, state_h_bwd_in, state_c_fwd_in, state_c_bwd_in], outputs=pred)
			else:
			# GRU
				recurrent_out = Bidirectional( REC_LAYER(n_recurrent_units, return_sequences=True, stateful=stateful) )(features_in, initial_state=[state_h_fwd_in, state_h_bwd_in])
				pred = Dense(1)(recurrent_out)
				test_model = keras.Model(inputs=[features_in, state_h_fwd_in, state_h_bwd_in], outputs=pred)
	else: # not bidi
		if not initialize_states:
			recurrent_out = REC_LAYER(n_recurrent_units, return_sequences=True, stateful=stateful )(features_in)
			pred = Dense(1)(recurrent_out)
			test_model = keras.Model(inputs=features_in, outputs=pred)
		else: 
			state_h_in = Input(batch_shape=(1,n_recurrent_units))
			state_c_in = Input(batch_shape=(1,n_recurrent_units))
			if layer_name == 'LSTM':
				recurrent_out = REC_LAYER(n_recurrent_units, return_sequences=True, stateful=stateful )(features_in, initial_state=[state_h_in, state_c_in])
				pred = Dense(1)(recurrent_out)
				test_model = keras.Model(inputs=[features_in, state_h_in, state_c_in], outputs=pred)
			else:
				# GRU
				recurrent_out = REC_LAYER(n_recurrent_units, return_sequences=True, stateful=stateful )(features_in, initial_state=state_h_in)
				pred = Dense(1)(recurrent_out)
				test_model = keras.Model(inputs=[features_in, state_h_in], outputs=pred)		
	test_model.compile(loss='mean_squared_error', optimizer='adam')
	if PRINT_SUMMARIES:
		test_model.summary()
	test_model.set_weights(weights)
	if bidi:
		model_fname = 'bidi-' + layer_name 
	else:
		model_fname = layer_name
	if stateful:
		model_fname += '_stateful'
	else:
		model_fname += '_nonstateful'
	if initialize_states:
		model_fname += '_init_state'
	else:
		model_fname += '_no_init_state'

	model_fname += '.h5'
	test_model.save(model_fname, include_optimizer=False)
	return test_model, model_fname

def eval_test_model(baseline_out, test_model, x_in, layer_name, bidi, stateful, states_initialized, initial_states=[]):
	num_test_seqs, sequence_length, feature_dim = x_in.shape
	results = np.zeros(0)
	for state_reset in [True, False]:
		for s in range(num_test_seqs):
			in_seq = x_in[s].reshape( (1, sequence_length, feature_dim) )
			baseline = baseline_out[s].reshape(sequence_length)
			msg = '\n\nRunning '
			if bidi:
				msg += 'Bidi-'
			msg += f'{layer_name}; Sequence {s}; stateful {stateful}; Initialzied state: {states_initialized}; State Reset: {state_reset}\n'
			if VERBOSE:
				print(msg)
			pred_in = x_in[s].reshape(x_inf[1:].shape)
			if not states_initialized:   # no initial state
				pred_seq = test_model.predict(pred_in)
			else:
				if layer_name == 'LSTM':
					if bidi:
						pred_seq = test_model.predict([pred_in, initial_states[0], initial_states[1], initial_states[2], initial_states[3]])
					else:
						pred_seq = test_model.predict([pred_in, initial_states[0], initial_states[1]])
				else: #GRU
					if bidi:
						pred_seq = test_model.predict([pred_in, initial_states[0], initial_states[1]])
					else:
						pred_seq = test_model.predict([pred_in, initial_states[0]])
			pred_seq = pred_seq.reshape(sequence_length)
			results = np.append(results, pred_seq)
			if VERBOSE:
				print(f'Baseline  : {baseline}')
				print(f'Prediction: {pred_seq}')
				print(f'Difference: {baseline - pred_seq}')
			if state_reset:
				if stateful and states_initialized:
					if VERBOSE:
						print('Keras does not handle reset_state calls with stateful=True with initial state inputs')
				else:
					test_model.reset_states()
	return results

##### generate toy data
train_seq_length = 4   
feature_dim = 1
num_seqs = 8

x_train = np.random.normal(0,1, num_seqs * train_seq_length * feature_dim )
x_train = x_train.reshape(num_seqs, train_seq_length, feature_dim)
y_train = np.random.normal(0,1, num_seqs * train_seq_length )
y_train = y_train.reshape(num_seqs, train_seq_length, 1)

n_recurrent_units = 2
all_results = np.zeros(0, dtype=np.float32)

# hand-generate the input data to make it easy to input into C++ code by hand.
# this hard-codes the train_seq_length, feature_dim vars for this purpose
x_inf = np.asarray( [ 2.1, -1.2, 3.14, 1.2, 1, 3, -2, 10 ], dtype=np.float32 ) # simple 
x_inf = x_inf.reshape( (2, train_seq_length, 1) )

initial_states = np.asarray( [ 1.1, -2.1, 2.7, 3.1, -2.5, 3.0, -2.0, -10.0 ], dtype=np.float32 )
initial_states = initial_states.reshape((4,1,2))

model_file_names = []
for bidi in [False, True]:
	for layer_name in ['GRU', 'LSTM']:
		### train with no initial state, no statefulness
		training_model, trained_weights = get_trained_model(x_train, y_train, layer_name, n_recurrent_units, bidi)
		y_inf = training_model.predict(x_inf)
		for stateful in [False, True]:
			for initialize_states in [False, True]:
				### evaluate the model 
				test_model, model_fname = get_test_model(n_recurrent_units, train_seq_length, feature_dim, layer_name, stateful, initialize_states, trained_weights, bidi)
				result = eval_test_model(y_inf, test_model, x_inf, layer_name, bidi, stateful, initialize_states, initial_states)
				all_results = np.append(all_results, result)
				model_file_names.append(model_fname)

if VERBOSE:
	print('\n\n')
	print(all_results)
	print(model_file_names)
