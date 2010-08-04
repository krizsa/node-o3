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
#ifndef O3_C_BLOB1_H
#define O3_C_BLOB1_H

namespace o3 {

struct cBlob1 : cScr {
    o3_begin_class(cScr)
    o3_end_class()

    o3_glue_gen()

    static o3_ext("cO3") o3_get siScr blob(iCtx* ctx)
    {
        o3_trace3 trace;
        Var blob = ctx->value("blob");

        if (blob.type() == Var::TYPE_VOID)
            blob = ctx->setValue("blob", o3_new(cBlob1)());
        return blob.toScr();
    }

    o3_fun Buf __self__(iCtx* ctx)
    {
        o3_trace3 trace;

        return Buf(ctx);
    }

    o3_fun Buf __self__(iCtx* ctx, size_t n)
    {
        o3_trace3 trace;
        Buf buf(n, ctx);

        buf.appendPattern((uint8_t) 0, n);
        return buf;
    }

    o3_fun Buf __self__(const Str& str)
    {
        o3_trace3 trace;

        return Buf(str);
    }

    o3_fun Buf fromString(const Str& str)
    {
        o3_trace3 trace;

        return Buf(str);
    }

    o3_fun Buf fromHex(const Str& str)
    {
        o3_trace3 trace;

        return Buf::fromHex(str.ptr(), str.alloc());
    }

    o3_fun Buf fromBase64(const Str& str)
    {
        o3_trace3 trace;

        return Buf::fromBase64(str.ptr(), str.alloc());
    }

    static o3_ext("cScrBuf") o3_fun Str toString(o3_tgt iScr* tgt)
    {
        o3_trace3 trace;
        cScrBuf* pthis = (cScrBuf*) tgt;
        Buf buf(pthis);

        return Str(buf);
    }

    static o3_ext("cScrBuf") o3_fun Str toHex(o3_tgt iScr* tgt)
    {
        o3_trace3 trace;
        cScrBuf* pthis = (cScrBuf*) tgt;
        Buf buf(pthis);

        return Str::fromHex(buf.ptr(), buf.size());
    }

    static o3_ext("cScrBuf") o3_fun Str toBase64(o3_tgt iScr* tgt)
    {
        o3_trace3 trace;
        cScrBuf* pthis = (cScrBuf*) tgt;
        Buf buf(pthis);

        return Str::fromBase64(buf.ptr(), buf.size());
    }
    
    static o3_ext("cScrBuf") o3_fun void replace(o3_tgt iScr* tgt, iBuf* orig,
        iBuf* rep) 
    {
        siBuf buf(tgt);
        Buf& orig_buf = orig->unwrap();
        Buf& replace_buf = rep->unwrap();

        if (!orig && !rep)
            return;

        buf->unwrap().findAndReplaceAll(orig_buf.ptr(), orig_buf.size(),
            replace_buf.ptr(), replace_buf.size());
    }

    static o3_ext("cScrBuf") o3_fun void replace(o3_tgt iScr* tgt, const char* orig,
        const char* rep)
    {    
        siBuf buf(tgt);

        buf->unwrap().findAndReplaceAll(orig, strLen(orig)*sizeof(char),
            rep, strLen(rep)*sizeof(char));
    }

    static o3_ext("cScrBuf") o3_fun void replaceUtf16(o3_tgt iScr* tgt, 
        const wchar_t* orig, const wchar_t* rep) 
    {
        siBuf buf(tgt);

        buf->unwrap().findAndReplaceAll(orig, strLen(orig)*sizeof(wchar_t),
            rep, strLen(rep)*sizeof(wchar_t));
    }
};    
}

#endif // O3_C_BLOB1_H
