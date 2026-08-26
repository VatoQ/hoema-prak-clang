#include "../include/dgl.h"

// TODO: implement
int derivatives(DGL_Result* target, const DGL* self, const Vector* y, double x)
{
    target->dgl_type = self->dgl_type;
    if (self->dgl_type == DGL_SYSTEM)
    {
        target->x_system = self->f_dgl_system(y, x);
        return DGL_SUCCESS;
    }
    else if (self->dgl_type == DGL_NTH_ORDER)
    {
        target->x_order = self->f_dgl_nth_order(y, x);

        return DGL_SUCCESS;
    }
    else
    {
        return DGL_BASIC_ERROR;
    }
}
