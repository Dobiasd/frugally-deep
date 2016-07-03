#include "size2d.h"

bool operator == (const size2d& lhs, const size2d& rhs)
{
    return
        lhs.height() == rhs.height() &&
        lhs.width() == rhs.width();
}
