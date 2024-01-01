/*
CUDA_VISIBLE_DEVICES='' python3.11 ../keras_export/generate_test_models.py exhaustive test_model_exhaustive.keras
CUDA_VISIBLE_DEVICES='' python3.11 ../keras_export/convert_model.py test_model_exhaustive.keras test_model_exhaustive.json
cat test_model_exhaustive.json | jq . > test_model_exhaustive.json.formatted.json
subl test_model_exhaustive.json.formatted.json
./applications_performance_d
*/

#include "fdeep/fdeep.hpp"
int main()
{
    fdeep::load_model("test_model_exhaustive.json");
}
