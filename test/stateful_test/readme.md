## Stand-Alone Stateful RNN Tests

This is a set of tests that is not part of the frugally-deep unit tests.  These test the functionality of the stateful property for RNNs (e.g., LSTMs, GRUs, and bidirectional variants).  This requires multiple calls to model.stateful_predict to test and that is why it is not covered by the standard unit tests.   

To run this test, just use the command `python3 stateful_recurrent_tests.py`.  This does the following:

1. Runs a series of RNN models in Keras and stores the models and some test vectors in the models subdirectory.  
2. Runs the `../../keras_export/convert_model.py ` scripy to convert all of the generated Keras models. The resulting json files also are written to the models subdirectory.  
3.  Compiles `stateful_recurrent_tests.cpp` which loads each of the json model files and generates the  test vectors corresponding to those in step 1 using frugally-deep.
4. Compares the test vectors from Keras and frugally-deep to report PASS/FAIL on the tests.  

The main Python script also creates the models directory as needed and clears it at the start of its execution.  