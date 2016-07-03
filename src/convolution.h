#pragma once

#include "filter.h"

#include <cstddef>
#include <vector>

matrix3d convolve(const std::vector<filter>& filters, const matrix3d& in_vol);
