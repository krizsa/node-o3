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
#ifndef O3_C_SCR_BUF_H
#define O3_C_SCR_BUF_H

namespace o3 {

o3_cls(cScrBuf);

struct cScrBuf : cScr, iBuf {
    Buf m_buf;

    cScrBuf(const Buf& buf = Buf()) : m_buf(buf)
    {
        o3_trace2 trace;
    }

    o3_begin_class(cScr)
        o3_add_iface(iBuf)
    o3_end_class()

	o3_glue_gen()

    o3_get size_t length()
    {
        o3_trace3 trace;

        return m_buf.size();
    }

    o3_set size_t setLength(size_t size)
    {
        o3_trace3 trace;

        m_buf.resize(size);
        return size;
    }

	o3_fun void append(const Buf& other)
	{
		m_buf.append(other.ptr(), other.size());
	}

	o3_fun Buf slice(size_t start, size_t end)
	{
		size_t size = m_buf.size();
		if(start<0 || end<0 || start>size || end>size || start>end)
			return Buf();

		return Buf(((int8_t*)m_buf.ptr())+start, end-start);
	}

    o3_fun size_t __enumerator__(size_t index)
    {
        o3_trace3 trace;

        if ((size_t) ++index < m_buf.size())
            return index;
        return (size_t) -1;
    }

    o3_fun bool __query__(size_t index)
    {
        o3_trace3 trace;

        return (size_t) index < m_buf.size();
    }

    o3_fun int  __getter__(size_t index)
    {
        o3_trace3 trace;

        if ((size_t) index < m_buf.size())
            return ((uint8_t*) m_buf.ptr())[index];
        return 0; 
    }

    o3_fun int __setter__(size_t index, int b)
    {
        o3_trace3 trace;

        if ((size_t) index < m_buf.size())
            return (int) (((uint8_t*) m_buf.ptr())[index] = (uint8_t)b);
        return 0; 
    }

    Buf& unwrap()
    {
        o3_trace2 trace;

        return m_buf;
    }
};

}

#endif // O3_C_SCR_BUF_H
