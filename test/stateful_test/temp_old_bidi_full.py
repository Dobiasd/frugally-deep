import h5py
import numpy as np
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional
from tensorflow.keras import Model

#### generate toy data
stream_seq_length = 4
seq_ratio = 2
train_seq_length = stream_seq_length * seq_ratio
feature_dim = 2
num_seqs = 3

x =  np.random.randint(0, high=2, size = (num_seqs * train_seq_length, feature_dim) )
x = np.sign( x - 0.5 )
y = np.sum( ( x == np.roll(x, 1, axis = 0) ), axis = 1 )
### y[n] = number of agreements between x[n], x[n-1]
x = x.reshape( (num_seqs, train_seq_length, feature_dim) )
y = y.reshape( (num_seqs, train_seq_length, 1) )


######  Define/Build/Train Training Model
lstm_output_dim = 4 

training_in_shape = x.shape[1:]
training_in = Input(shape=training_in_shape)
# training_in = Input(batch_shape=(None,train_seq_length,feature_dim)) this works too
foo = Bidirectional( LSTM(lstm_output_dim, return_sequences=True, stateful=False) ) (training_in)
training_pred = Dense(1)(foo)
training_model = Model(inputs=training_in, outputs=training_pred)
training_model.compile(loss='mean_squared_error', optimizer='adam')
training_model.summary()
training_model.fit(x, y, batch_size=2, epochs=10)

training_model.save_weights('weights.hd5', overwrite=True)

##### define the streaming-infernece model
streaming_in = Input(batch_shape=(1,stream_seq_length,feature_dim))  ## stateful ==> needs batch_shape specified
initial_state_h = Input(batch_shape=(1,lstm_output_dim))
initial_state_c = Input(batch_shape=(1,lstm_output_dim))

fwd_initial_state_h = Input(batch_shape=(1,lstm_output_dim))
fwd_initial_state_c = Input(batch_shape=(1,lstm_output_dim))
bwd_initial_state_h = Input(batch_shape=(1,lstm_output_dim))
bwd_initial_state_c = Input(batch_shape=(1,lstm_output_dim))


lstm_streaming_stateful = Bidirectional( LSTM(lstm_output_dim, return_sequences=True, stateful=True ) ) (streaming_in)
lstm_streaming_initial_stateful = Bidirectional(  LSTM(lstm_output_dim, return_sequences=True, stateful=True) ) (streaming_in, initial_state=[fwd_initial_state_h, fwd_initial_state_c, bwd_initial_state_h, bwd_initial_state_c])
lstm_streaming_initial_nonstateful = Bidirectional( LSTM(lstm_output_dim, return_sequences=True, stateful=False) ) (streaming_in, initial_state=[fwd_initial_state_h, fwd_initial_state_c, bwd_initial_state_h, bwd_initial_state_c])


streaming_stateful_model = Model( inputs=streaming_in, outputs=Dense(1)(lstm_streaming_stateful) )
streaming_initial_stateful_model = Model( inputs=[streaming_in,fwd_initial_state_h, fwd_initial_state_c, bwd_initial_state_h, bwd_initial_state_c], outputs=Dense(1)(lstm_streaming_initial_stateful) )
streaming_initial_nonstateful_model = Model( inputs=[streaming_in, fwd_initial_state_h, fwd_initial_state_c, bwd_initial_state_h, bwd_initial_state_c], outputs=Dense(1)(lstm_streaming_initial_nonstateful) )

streaming_stateful_model.compile(loss='mean_squared_error', optimizer='adam')
streaming_initial_stateful_model.compile(loss='mean_squared_error', optimizer='adam')
streaming_initial_nonstateful_model.compile(loss='mean_squared_error', optimizer='adam')

streaming_stateful_model.load_weights('weights.hd5')
streaming_initial_stateful_model.load_weights('weights.hd5')
streaming_initial_nonstateful_model.load_weights('weights.hd5')


##### demo the behaivor

fwd_initial_c = np.random.normal(0,1, (1,lstm_output_dim))
fwd_initial_h = np.random.normal(0,1, (1,lstm_output_dim))
bwd_initial_c = np.random.normal(0,1, (1,lstm_output_dim))
bwd_initial_h = np.random.normal(0,1, (1,lstm_output_dim))
# initial_c = np.zeros((1,lstm_output_dim))
# initial_h = np.zeros((1,lstm_output_dim))

state_reset = True

print('\n\n******the streaming-inference model can replicate the sequence-based trained model:\n')
for s in range(num_seqs):
	if state_reset:
		print(f'\n\nRunning Sequence {s} with STATE RESET:\n')
	else:
		print(f'\n\nRunning Sequence {s} with NO STATE RESET:\n')
	in_seq = x[s].reshape( (1, train_seq_length, feature_dim) )
	seq_pred = training_model.predict(in_seq)
	seq_pred = seq_pred.reshape(train_seq_length)
	for k in range(seq_ratio):
		in_feature_vector = x[s][ k * stream_seq_length : (k+1) * stream_seq_length ].reshape(1,stream_seq_length,feature_dim)
		stream_stateful_pred = streaming_stateful_model.predict(in_feature_vector).reshape(stream_seq_length)
		stream_stateful_init_pred = streaming_initial_stateful_model.predict([in_feature_vector, fwd_initial_c, fwd_initial_h, bwd_initial_c, bwd_initial_h]).reshape(stream_seq_length)
		stream_nonstateful_init_pred = streaming_initial_nonstateful_model.predict([in_feature_vector, fwd_initial_c, fwd_initial_h, bwd_initial_c, bwd_initial_h]).reshape(stream_seq_length)
		seq_pred_chunk = seq_pred[k * stream_seq_length : (k+1) * stream_seq_length]
		for n in range(stream_seq_length):
			msg = f'Seq-model Prediction[{k}, {n}]: { seq_pred_chunk[n] : 3.2f}\n'
			msg += f'no-init-stateful, diff: {stream_stateful_pred[n] : 3.2f}, {seq_pred_chunk[n] - stream_stateful_pred[n] : 3.2f}\n'
			msg += f'init-stateful, diff: {stream_stateful_init_pred[n] : 3.2f}, {seq_pred_chunk[n] - stream_stateful_init_pred[n] : 3.2f}\n'
			msg += f'init-not-stateful, diff: {stream_nonstateful_init_pred[n] : 3.2f}, {seq_pred_chunk[n] - stream_nonstateful_init_pred[n] : 3.2f}\n\n'
			print(msg)
	if state_reset:
		streaming_stateful_model.reset_states()
		# streaming_initial_stateful_model.reset_states()  <<----- this causes an error.  
		streaming_initial_nonstateful_model.reset_states()
