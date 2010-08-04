/*
 * Copyright (C) 2010 Ajax.org BV
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 2 of the License, or (at your option) any later
 * version.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this library; if not, write to the Free Software Foundation, Inc., 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */
#ifndef O3_C_SCREEN1_WIN32_BASE_H
#define O3_C_SCREEN1_WIN32_BASE_H

namespace o3 {

o3_cls(cScreen1);

struct cScreen1: cScreen1Base
{
    cScreen1()
    {}

    virtual ~cScreen1()
    {}

    o3_begin_class(cScreen1Base)
    o3_end_class()

    o3_glue_gen()

    o3_ext("cO3") o3_get static siScr screen(iCtx* ctx)
    {
        Var v = ctx->value("screen");
        siScr ret = v.toScr();
        if (ret)
            return ret;

        ret = o3_new(cScreen1)();
        v = ret;
        ctx->setValue("screen", v);    
    
        return ret;
    }

    o3_get int width()
    {
        return GetSystemMetrics(SM_CXSCREEN);
    }

    o3_get int height()
    {
        return GetSystemMetrics(SM_CYSCREEN);
    }

};

}

#endif // O3_C_SCREEN1_WIN32_BASE_H
