message(STATUS "Building Unit Tests ${UNITTEST}")

add_custom_command ( OUTPUT test_model_exhaustive.h5
                     COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/keras_export/generate_test_models.py exhaustive test_model_exhaustive.h5"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_embedding.h5
                     COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/keras_export/generate_test_models.py embedding test_model_embedding.h5"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_recurrent.h5
                     COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/keras_export/generate_test_models.py recurrent test_model_recurrent.h5"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_lstm.h5
                     COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/keras_export/generate_test_models.py lstm test_model_lstm.h5"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_lstm_stateful.h5
                     COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/keras_export/generate_test_models.py lstm_stateful test_model_lstm_stateful.h5"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_gru.h5
                     COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/keras_export/generate_test_models.py gru test_model_gru.h5"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_gru_stateful.h5
                     COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/keras_export/generate_test_models.py gru_stateful test_model_gru_stateful.h5"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_variable.h5
                     COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/keras_export/generate_test_models.py variable test_model_variable.h5"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_sequential.h5
                     COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/keras_export/generate_test_models.py sequential test_model_sequential.h5"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT readme_example_model.h5
                     COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/test/readme_example_generate.py"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_exhaustive.json
                     DEPENDS test_model_exhaustive.h5
                     COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/keras_export/convert_model.py test_model_exhaustive.h5 test_model_exhaustive.json"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_embedding.json
                     DEPENDS test_model_embedding.h5
                     COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/keras_export/convert_model.py test_model_embedding.h5 test_model_embedding.json"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_recurrent.json
                     DEPENDS test_model_recurrent.h5
                     COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/keras_export/convert_model.py test_model_recurrent.h5 test_model_recurrent.json"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_lstm.json
                     DEPENDS test_model_lstm.h5
                     COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/keras_export/convert_model.py test_model_lstm.h5 test_model_lstm.json"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_lstm_stateful.json
                     DEPENDS test_model_lstm_stateful.h5
                     COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/keras_export/convert_model.py test_model_lstm_stateful.h5 test_model_lstm_stateful.json"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_gru.json
                     DEPENDS test_model_gru.h5
                     COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/keras_export/convert_model.py test_model_gru.h5 test_model_gru.json"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_gru_stateful.json
                     DEPENDS test_model_gru_stateful.h5
                     COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/keras_export/convert_model.py test_model_gru_stateful.h5 test_model_gru_stateful.json"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_variable.json
                     DEPENDS test_model_variable.h5
                     COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/keras_export/convert_model.py test_model_variable.h5 test_model_variable.json"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_sequential.json
                     DEPENDS test_model_sequential.h5
                     COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/keras_export/convert_model.py test_model_sequential.h5 test_model_sequential.json"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT readme_example_model.json
                     DEPENDS readme_example_model.h5
                     COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/keras_export/convert_model.py readme_example_model.h5 readme_example_model.json"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

hunter_add_package(doctest)
find_package(doctest CONFIG REQUIRED)

macro(_add_test _NAME _DEPENDS)
    add_custom_target(${_NAME}_data DEPENDS ${_DEPENDS})
    add_executable(${_NAME} ${_NAME}.cpp)
    add_dependencies(${_NAME} ${_NAME}_data)
    add_test(NAME ${_NAME} COMMAND ${_NAME})
    target_link_libraries(${_NAME} fdeep Threads::Threads doctest::doctest)
endmacro()

_add_test(test_model_exhaustive_test test_model_exhaustive.json)
_add_test(test_model_embedding_test test_model_embedding.json)
_add_test(test_model_recurrent_test test_model_recurrent.json)
_add_test(test_model_lstm_test test_model_lstm.json)
_add_test(test_model_lstm_stateful_test test_model_lstm_stateful.json)
_add_test(test_model_gru_test test_model_gru.json)
_add_test(test_model_gru_stateful_test test_model_gru_stateful.json)
_add_test(test_model_variable_test test_model_variable.json)
_add_test(test_model_sequential_test test_model_sequential.json)
_add_test(readme_example_main readme_example_model.json)

add_custom_target(unittest
  COMMAND test_model_exhaustive_test
  COMMAND test_model_embedding_test
  COMMAND test_model_recurrent_test
  COMMAND test_model_lstm_test
  COMMAND test_model_lstm_stateful_test
  COMMAND test_model_gru_test
  COMMAND test_model_gru_stateful_test
  COMMAND test_model_variable_test
  COMMAND test_model_sequential_test
  COMMAND readme_example_main

  COMMENT "Running unittests\n\n"
  VERBATIM
)
