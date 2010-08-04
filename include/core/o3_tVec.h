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
#ifndef O3_T_VEC_H
#define O3_T_VEC_H

namespace o3 {

template<typename T>
class tVec {
    Buf m_buf;

public:
    explicit tVec(size_t capacity, iAlloc* alloc = g_sys) :
        m_buf(capacity * sizeof(T), alloc)
    {
        o3_trace1 trace;
    }

    explicit tVec(iAlloc* alloc = g_sys)
    {
        o3_trace1 trace;

        new(this) tVec(0, alloc);
    }

	tVec(const tVec& that)
	{
		if (that.m_buf.capacity() <= O3_AUTO_CAPACITY) {
			new(this) tVec(that.capacity(), that.alloc());

			append(that.ptr(), that.size());
		} else
			m_buf = that.m_buf;
	}

	tVec& operator=(const tVec& that)
	{
		o3_trace1 trace;

		if (this != &that) {
			tVec tmp(that);

			swap(*this, tmp);
		}
		return *this;
	}

    ~tVec()
    {
        o3_trace1 trace;

        if (m_buf.unique())
            clear();
    }

    bool empty() const
    {
        o3_trace1 trace;

        return size() == 0;
    }

    iAlloc* alloc() const
    {
        o3_trace1 trace;

        return m_buf.alloc();
    }

    size_t capacity() const
    {
        o3_trace1 trace;

        return m_buf.capacity() / sizeof(T);
    }

    size_t size() const
    {
        o3_trace1 trace;

        return m_buf.size() / sizeof(T);
    }

    const T* ptr() const
    {
        o3_trace1 trace;

        return (const T*) m_buf.ptr();
    }

    operator const T*() const
    {
        return ptr();
    }

    const T& back() const
    {
        return ptr()[size() - 1];
    }

    T* ptr()
    {
        o3_trace1 trace;

        if (!m_buf.unique()) {
            tVec tmp(capacity(), alloc());

            tmp.append(((const tVec*) this)->ptr(), size());
            swap(*this, tmp);
        }
        return (T*) m_buf.ptr();
    }

    operator T*()
    {
        return ptr();
    }

    T& back()
    {
        return ptr()[size() - 1];
    }

    void insert(size_t pos, const T& x, size_t n = 1)
    {
        o3_trace1 trace;
        T* ptr;

        m_buf.shift(pos * sizeof(T), n * sizeof(T));
        ptr = this->ptr() + pos;
        while (n--)
            new(ptr++) T(x);
    }

    void insert(size_t pos, const T* ptr, size_t n)
    {
        o3_trace1 trace;
        T* ptr1;

        m_buf.shift(pos * sizeof(T), n * sizeof(T));
        ptr1 = this->ptr() + pos;
        while (n--)
            new(ptr1++) T(*ptr++);
    }

    void append(const T& x, size_t n = 1)
    {
        o3_trace1 trace;

        insert(size(), x, n);
    }

    void append(const T* ptr, size_t n)
    {
        o3_trace1 trace;

        insert(size(), ptr, n);
    }

    void remove(size_t pos, size_t n = 1)
    {
        o3_trace1 trace;
        T* ptr = this->ptr();

        for (size_t i = n; i > 0; --i)
            ptr++->~T();
        m_buf.remove(pos * sizeof(T), n * sizeof(T));
    }

    void clear()
    {
        o3_trace1 trace;

        remove((size_t) 0, size());
    }

    void push(const T& x)
    {
        o3_trace1 trace;

        append(x);
    }

    void pop()
    {
        o3_trace1 trace;

        remove(size() - 1);
    }
};

}

#endif // O3_T_VEC_H
