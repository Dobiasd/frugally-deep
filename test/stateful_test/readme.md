# Stand-Alone Stateful RNN Tests

This is a set of tests that is not part of the frugally-deep unit tests.  These test the functionality of the stateful property for RNNs (e.g., LSTMs, GRUs, and bidirectional variants).  This requires multiple calls to `model.stateful_predict()` to test and that is why it is not covered by the standard unit tests which only verify test vectors for a single call to `model.predict()` or `model.stateful_predict()`.

To execute this test, just run the following commands:

```bash
mkdir models
g++ -I../../include -std=c++14 -O3 stateful_recurrent_tests.cpp -o stateful_recurrent_tests_cpp
python3 stateful_recurrent_tests.py
rm -rf models
```

This does the following:

1. Runs a series of RNN models in Keras and stores the models and some test vectors in the `models` subdirectory.
2. Runs the `../../keras_export/convert_model.py ` script to convert all of the generated Keras models. The resulting JSON files also are written to the `models` subdirectory.
3. Compiles and runs `stateful_recurrent_tests.cpp` which loads each of the JSON model files and generates test vectors using frugally-deep that correspond to those in step 1.
4. Compares the test vectors from Keras and frugally-deep to report PASS/FAIL on the tests.
