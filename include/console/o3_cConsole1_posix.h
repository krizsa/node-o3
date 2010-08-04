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
#ifndef O3_C_CONSOLE1_POSIX_H
#define O3_C_CONSOLE1_POSIX_H

namespace o3 {

struct cConsole1 : cScr {

    o3_begin_class(cScr)
    o3_end_class()

    o3_glue_gen()

    static o3_ext("cO3") o3_get siStream stdIn(iCtx* ctx)
    {
        o3_trace3 trace;
        Var in = ctx->value("in");

        if (in.type() == Var::TYPE_VOID)
            in = ctx->setValue("in", o3_new(cStream)(stdin));
        return in.toScr();
    }

    static o3_ext("cO3") o3_get siStream stdOut(iCtx* ctx)
    {
#ifdef O3_WIN32
        static HANDLE stdout = GetStdHandle(STD_OUTPUT_HANDLE);
#endif // O3_WIN32
        o3_trace3 trace;
        Var out = ctx->value("out");
        if (out.type() == Var::TYPE_VOID)
            out = ctx->setValue("out", o3_new(cStream)(stdout));
		return out.toScr();
    }

    static o3_ext("cO3") o3_get siStream stdErr(iCtx* ctx)
    {
#ifdef O3_WIN32
        static HANDLE stdout = GetStdHandle(STD_ERROR_HANDLE);
#endif // O3_WIN32
        o3_trace3 trace;
        Var err = ctx->value("err");

        if (err.type() == Var::TYPE_VOID)
            err = ctx->setValue("err", o3_new(cStream)(stderr));
        return err.toScr();
    }

    static o3_ext("cO3") o3_fun void print(iCtx* ctx, const Str& str)
    {		
        o3_trace3 trace;
        siStream out = cConsole1::stdOut(ctx);		
        out->write(str.ptr(), str.size());
        out->flush();
    }
};

}

#endif // O3_C_CONSOLE1_POSIX_H
