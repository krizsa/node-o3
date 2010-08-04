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
#ifndef O3_CSCREEN1_APPLE_H
#define O3_CSCREEN1_APPLE_H

#include <Cocoa/Cocoa.h>

namespace o3 {

struct cScreen1 : cScreen1Base {
    static o3_ext("cO3") o3_fun siScr screen(iCtx* ctx)
    {
        static Var screen = ctx->value("screen");

        if (screen.type() == Var::TYPE_VOID)
            screen = ctx->setValue("screen", o3_new(cScreen1)());
        return screen.toScr();
    }

    o3_begin_class(cScreen1Base)
    o3_end_class()

    o3_glue_gen()

    int width()
    {
        o3_trace3 trace;

        return [[NSScreen mainScreen] frame].size.width;
    }

    int height()
    {
        o3_trace3 trace;

        return [[NSScreen mainScreen] frame].size.height;
    }
};

}

#endif // O3_CSCREEN1_APPLE_H
