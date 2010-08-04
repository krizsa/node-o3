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
#ifndef O3_T_SCR_VEC_H
#define O3_T_SCR_VEC_H

namespace o3 {

template<typename T>
struct tScrVec : cScr {
    tVec<T> m_elems;
    size_t  m_length;
    tMap<size_t, Var> m_vals;

public:
    tScrVec(const tVec<T>& elems) : m_elems(elems), m_length(elems.size())
    {
        o3_trace2 trace;
    }

    o3_begin_class(cScr)
    o3_end_class()

#include "o3_tScrVec_scr.h"

    o3_get size_t length()
    {
        o3_trace3 trace;

        return m_length;
    }

    o3_set size_t setLength(size_t length)
    {
        o3_trace3 trace;

        length = max(m_elems.size(), length);
        while (m_length-- > length)
            __deleter__(m_length);
        return m_length;
    }

    o3_fun size_t __enumerator__(size_t index)
    {
        o3_trace3 trace;

        if (++index < m_elems.size())
            return index;
        for (tMap<size_t, Var>::ConstIter i = m_vals.begin(); i != m_vals.end();
             ++i)
            if (i->key >= index)
                return i->key;
        return NOT_FOUND;
    }

    o3_fun bool __query__(size_t index)
    {
        o3_trace3 trace;

        return index < m_elems.size() || m_vals.find(index) != m_vals.end();
    }

    o3_fun Var __getter__(iCtx* ctx, size_t index)
    {
        o3_trace3 trace;
        tMap<size_t, Var>::ConstIter iter = m_vals.find(index);

        if (iter != m_vals.end())
            return iter->val;
        if (index < m_elems.size()) 
            return m_vals[index] = Var(m_elems[index], ctx);
        return Var(ctx);
    }

    o3_fun Var __setter__(size_t index, const Var& val)
    {
        o3_trace3 trace;

        return m_vals[index] = val;
    }

    o3_fun bool __deleter__(size_t index)
    {
        o3_trace3 trace;
        tMap<size_t, Var>::Iter iter = m_vals.find(index);

        if (iter != m_vals.end()) {
            m_vals.remove(iter);
            return true;
        }
        return false;
    }
};

}

#endif // O3_T_SCR_VEC_H
