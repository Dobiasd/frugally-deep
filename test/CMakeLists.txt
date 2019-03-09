message(STATUS "Building Unit Tests ${UNITTEST}")

set(full_test_default OFF)

option(FDEEP_BUILD_FULL_TEST "Build full model tests" ${full_test_default})

add_custom_command ( OUTPUT test_model_small.h5
                     COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/keras_export/generate_test_models.py small test_model_small.h5"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_pooling.h5
                     COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/keras_export/generate_test_models.py pooling test_model_pooling.h5"
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

add_custom_command ( OUTPUT test_model_gru.h5
                     COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/keras_export/generate_test_models.py gru test_model_gru.h5"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_variable.h5
                     COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/keras_export/generate_test_models.py variable test_model_variable.h5"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_sequential.h5
                     COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/keras_export/generate_test_models.py sequential test_model_sequential.h5"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

if(FDEEP_BUILD_FULL_TEST)
    add_custom_command ( OUTPUT test_model_full.h5
                         COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/keras_export/generate_test_models.py full test_model_full.h5"
                         WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)
endif()

add_custom_command ( OUTPUT readme_example_model.h5
                     COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/test/readme_example_generate.py"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_small.json
                     DEPENDS test_model_small.h5
                     COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/keras_export/convert_model.py test_model_small.h5 test_model_small.json"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_pooling.json
                     DEPENDS test_model_pooling.h5
                     COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/keras_export/convert_model.py test_model_pooling.h5 test_model_pooling.json"
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

add_custom_command ( OUTPUT test_model_gru.json
                     DEPENDS test_model_gru.h5
                     COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/keras_export/convert_model.py test_model_gru.h5 test_model_gru.json"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_variable.json
                     DEPENDS test_model_variable.h5
                     COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/keras_export/convert_model.py test_model_variable.h5 test_model_variable.json"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_sequential.json
                     DEPENDS test_model_sequential.h5
                     COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/keras_export/convert_model.py test_model_sequential.h5 test_model_sequential.json"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

if(FDEEP_BUILD_FULL_TEST)
    add_custom_command ( OUTPUT test_model_full.json
                         DEPENDS test_model_full.h5
                         COMMAND bash -c "python3 ${FDEEP_TOP_DIR}/keras_export/convert_model.py test_model_full.h5 test_model_full.json"
                         WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)
endif()

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

_add_test(test_model_small_test test_model_small.json)
_add_test(test_model_pooling_test test_model_pooling.json)
_add_test(test_model_embedding_test test_model_embedding.json)
_add_test(test_model_recurrent_test test_model_recurrent.json)
_add_test(test_model_lstm_test test_model_lstm.json)
_add_test(test_model_gru_test test_model_gru.json)
_add_test(test_model_variable_test test_model_variable.json)
_add_test(test_model_sequential_test test_model_sequential.json)
if(FDEEP_BUILD_FULL_TEST)
  _add_test(test_model_full_test test_model_full.json)
  _add_test(test_model_full_test_double test_model_full.json)
endif()
_add_test(readme_example_main readme_example_model.json)

if(FDEEP_BUILD_FULL_TEST)
  add_custom_target(unittest
    COMMAND test_model_small_test
    COMMAND test_model_pooling_test
    COMMAND test_model_embedding_test
    COMMAND test_model_recurrent_test
    COMMAND test_model_lstm_test
    COMMAND test_model_gru_test
    COMMAND test_model_variable_test
    COMMAND test_model_sequential_test
    COMMAND test_model_full_test
    COMMAND test_model_full_test_double
    COMMAND readme_example_main

    COMMENT "Running unittests\n\n"
    VERBATIM
    )
else()
  add_custom_target(unittest
    COMMAND test_model_small_test
    COMMAND test_model_pooling_test
    COMMAND test_model_embedding_test
    COMMAND test_model_recurrent_test
    COMMAND test_model_lstm_test
    COMMAND test_model_gru_test
    COMMAND test_model_variable_test
    COMMAND test_model_sequential_test
    COMMAND readme_example_main

    COMMENT "Running unittests\n\n"
    VERBATIM
    )
endif()
