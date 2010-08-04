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
#ifndef O3_C_STREAM_BASE_H
#define O3_C_STREAM_BASE_H

namespace o3 {

o3_cls(cStreamBase);

struct cStreamBase : cScr, iStream {
    o3_begin_class(cScr)
        o3_add_iface(iStream)
    o3_end_class();

	o3_glue_gen()

    virtual o3_get bool eof() = 0;

    virtual o3_get bool error() = 0;

    virtual o3_get size_t pos() = 0;

    virtual o3_set size_t setPos(size_t size) = 0;

    virtual size_t read(void* data, size_t size) = 0;

    virtual o3_fun Buf readBlob(size_t n)
    {
        o3_trace3 trace;

        return Buf(this, n);
    }

    virtual o3_fun Str read(size_t n)
    {
        o3_trace3 trace;

        return Str(readBlob(n * sizeof(char)));
    }

    virtual size_t write(const void* data, size_t size) = 0;

    virtual o3_fun size_t write(const Buf& buf)
    {
        o3_trace3 trace;

        return write(buf.ptr(), buf.size());
    }

    virtual o3_fun size_t write(const Str& str)
    {
        o3_trace3 trace;

        return write(str.ptr(), str.size());
    }

    virtual o3_fun bool flush() = 0;

    virtual o3_fun bool close() = 0;

    o3_fun void print(const Str& str)
    {
        o3_trace3 trace;

        write(str.ptr(), str.size());
        flush();
    }
};

}

#endif // O3_C_STREAM_BASE_H
