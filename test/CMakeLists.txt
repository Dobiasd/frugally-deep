message(STATUS "Building Unit Tests ${UNITTEST}")

# look for Python3
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
 # Please make sure that Python3 and Tensorflow have been installed correctly.
message(STATUS "Please make sure Tensorflow is installed.")
add_custom_command ( OUTPUT test_model_exhaustive.keras
                     COMMAND bash -c "${Python3_EXECUTABLE} ${FDEEP_TOP_DIR}/keras_export/generate_test_models.py exhaustive test_model_exhaustive.keras"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_embedding.keras
                     COMMAND bash -c "${Python3_EXECUTABLE} ${FDEEP_TOP_DIR}/keras_export/generate_test_models.py embedding test_model_embedding.keras"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_variable.keras
                     COMMAND bash -c "${Python3_EXECUTABLE} ${FDEEP_TOP_DIR}/keras_export/generate_test_models.py variable test_model_variable.keras"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_autoencoder.keras
                     COMMAND bash -c "${Python3_EXECUTABLE} ${FDEEP_TOP_DIR}/keras_export/generate_test_models.py autoencoder test_model_autoencoder.keras"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_sequential.keras
                     COMMAND bash -c "${Python3_EXECUTABLE} ${FDEEP_TOP_DIR}/keras_export/generate_test_models.py sequential test_model_sequential.keras"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT readme_example_model.keras
                     COMMAND bash -c "${Python3_EXECUTABLE} ${FDEEP_TOP_DIR}/test/readme_example_generate.py"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_exhaustive.json
                     DEPENDS test_model_exhaustive.keras
                     COMMAND bash -c "${Python3_EXECUTABLE} ${FDEEP_TOP_DIR}/keras_export/convert_model.py test_model_exhaustive.keras test_model_exhaustive.json"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_embedding.json
                     DEPENDS test_model_embedding.keras
                     COMMAND bash -c "${Python3_EXECUTABLE} ${FDEEP_TOP_DIR}/keras_export/convert_model.py test_model_embedding.keras test_model_embedding.json"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_variable.json
                     DEPENDS test_model_variable.keras
                     COMMAND bash -c "${Python3_EXECUTABLE} ${FDEEP_TOP_DIR}/keras_export/convert_model.py test_model_variable.keras test_model_variable.json"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_autoencoder.json
                     DEPENDS test_model_autoencoder.keras
                     COMMAND bash -c "${Python3_EXECUTABLE} ${FDEEP_TOP_DIR}/keras_export/convert_model.py test_model_autoencoder.keras test_model_autoencoder.json"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT test_model_sequential.json
                     DEPENDS test_model_sequential.keras
                     COMMAND bash -c "${Python3_EXECUTABLE} ${FDEEP_TOP_DIR}/keras_export/convert_model.py test_model_sequential.keras test_model_sequential.json"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_custom_command ( OUTPUT readme_example_model.json
                     DEPENDS readme_example_model.keras
                     COMMAND bash -c "${Python3_EXECUTABLE} ${FDEEP_TOP_DIR}/keras_export/convert_model.py readme_example_model.keras readme_example_model.json"
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
_add_test(test_model_variable_test test_model_variable.json)
_add_test(test_model_autoencoder_test test_model_autoencoder.json)
_add_test(test_model_sequential_test test_model_sequential.json)
_add_test(readme_example_main readme_example_model.json)

add_custom_target(unittest
  COMMAND test_model_exhaustive_test
  COMMAND test_model_embedding_test
  COMMAND test_model_variable_test
  COMMAND test_model_autoencoder_test
  COMMAND test_model_sequential_test
  COMMAND readme_example_main

  COMMENT "Running unittests\n\n"
  VERBATIM
)
