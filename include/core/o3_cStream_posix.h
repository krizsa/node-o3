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
#ifndef O3_C_STREAM_POSIX_H
#define O3_C_STREAM_POSIX_H

#include <stdio.h>

namespace o3 {

o3_cls(cStream);

struct cStream : cStreamBase {
    FILE* m_stream;

    cStream(FILE* stream)
    {
        o3_trace2 trace;

        m_stream = stream;
    }

    virtual ~cStream()
    {
        o3_trace2 trace;

        if (m_stream)
            ::fclose(m_stream);
    }

    o3_begin_class(cStreamBase)
    o3_end_class()

	o3_glue_gen()

    bool eof()
    {
        o3_trace3 trace;

        return ::feof(m_stream);
    }

    bool error()
    {
        o3_trace3 trace;

        return ::ferror(m_stream);
    }

    size_t pos()
    {
        o3_trace3 trace;

        return ::ftell(m_stream);
    }

    size_t setPos(size_t pos)
    {
        o3_trace3 trace;

        ::fseek(m_stream, pos, SEEK_SET);
        return pos;
    }

    size_t read(void* ptr, size_t size)
    {
        o3_trace2 trace;

        return ::fread(ptr, sizeof(uint8_t), size, m_stream);
    }

    size_t write(const void* ptr, size_t size)
    {
        o3_trace2 trace;
        
        return ::fwrite(ptr, sizeof(uint8_t), size, m_stream);
    }

    bool flush()
    {
        o3_trace3 trace;

        return ::fflush(m_stream) == 0;
    }

    bool close()
    {
        o3_trace3 trace;

        return ::fclose(m_stream) == 0;
    }

    void* unwrap()
    {
        o3_trace2 trace;

        return m_stream;
    }

	size_t size()
	{
		// TODO: implement
		o3_assert(false);
		return 0;
	}
};

}

#endif // O3_C_STREAM_POSIX_H
