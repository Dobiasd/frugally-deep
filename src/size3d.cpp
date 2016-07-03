#include "size3d.h"

bool operator == (const size3d& lhs, const size3d& rhs)
{
    return
        lhs.depth() == rhs.depth() &&
        lhs.height() == rhs.height() &&
        lhs.width() == rhs.width();
}
