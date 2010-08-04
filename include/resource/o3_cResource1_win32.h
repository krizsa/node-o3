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
#ifndef O3_C_RESOURCE_1_WIN32_H
#define O3_C_RESOURCE_1_WIN32_H

namespace o3{

struct cResource1 : cScr
{
    cResource1()
    {}

    virtual ~cResource1()
    {}

    o3_begin_class(cScr)
    o3_end_class()

    o3_glue_gen()

    // get the resource comp singleton, that handles appended resource files
    static o3_ext("cO3") o3_get siScr resources(iCtx* ctx)
    {
        Var v = ctx->value("resources");
        siScr ret = v.toScr();
        if (ret)
            return ret;

        cResource1* rsc = o3_new(cResource1)();
        ret = rsc;

        v = ret;
        ctx->setValue("resources", v);
        return ret;    
    }

    // unpack all resource files to the dir/file pointed by 'to'
    // (the dir structure will be the same)
    o3_fun void unpack(iCtx* ctx, iFs* fs, siEx* ex)
    {
        ctx; fs; ex;
    }

    // returns a list with the appended files
    virtual o3_fun tVec<Str> list()
    {
        return ((cSys*) g_sys)->resourcePaths();
    }

    // get the data from an appended file specified by the path
    o3_fun Buf get(const Str& path)
    {
        return ((cSys*) g_sys)->resource(path);
    }

    o3_fun siStream protocolOpen(const char* path)
    {
        Buf data = ((cSys*) g_sys)->resource(path);
        return o3_new(cBufStream)(data);
    }
};
}

#endif // O3_C_RESOURCE1_WIN32_H
