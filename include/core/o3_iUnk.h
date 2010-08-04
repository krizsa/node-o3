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
#ifndef O3_I_UNK_H
#define O3_I_UNK_H

#define o3_iid(T, l, s1, s2, c1, c2, c3, c4, c5, c6, c7, c8) \
struct T; \
typedef tSi<T> s##T; \
\
inline const Guid& iidof(T*) \
{ \
    o3_trace0 trace; \
    static const Guid IID = { \
        l, \
        s1, \
        s2, \
        { c1, c2, c3, c4, c5, c6, c7, c8 } \
    }; \
    \
    return IID; \
}

#define o3_cast (iUnk*) (void*)

namespace o3 {

struct Guid {
    uint32_t data1;
    uint16_t data2;
    uint16_t data3;
    uint8_t data4[8];
};

inline bool operator==(const Guid& x, const Guid& y)
{
    o3_trace0 trace;

    return ((uint32_t*) &x)[0] == ((uint32_t*) &y)[0]
        && ((uint32_t*) &x)[1] == ((uint32_t*) &y)[1]
        && ((uint32_t*) &x)[2] == ((uint32_t*) &y)[2]    
        && ((uint32_t*) &x)[3] == ((uint32_t*) &y)[3];
}

inline bool operator!=(const Guid& x, const Guid& y)
{
    o3_trace0 trace;

    return !(x == y);
}

struct iUnk {
    virtual int32_t queryInterface(const Guid& iid, void** obj) = 0;

    virtual uint32_t addRef() = 0;

    virtual uint32_t release() = 0;
};

template<typename T>
class tSi {
    template<typename T1>
    friend class tSi;

    T* m_ptr;
 
public:
    tSi(T* ptr = 0) : m_ptr(ptr)
    {
        o3_trace1 trace;

        if (m_ptr)
            m_ptr->addRef();
    }

    template<typename T1>
    tSi(T1* ptr) : m_ptr(0)
    {
        o3_trace1 trace;

        if (ptr)
            (o3_cast ptr)->queryInterface(iidof(m_ptr), (void**) &m_ptr);
    }

    tSi(const tSi& that) : m_ptr(that.m_ptr)
    {
        o3_trace1 trace;

        if (m_ptr)
            m_ptr->addRef();
    }

    template<typename T1>
    tSi(const tSi<T1>& that) : m_ptr(0)
    {
        o3_trace1 trace;

        if (that.m_ptr)
            that.m_ptr->queryInterface(iidof(m_ptr), (void**) &m_ptr);
    }

    tSi& operator=(const tSi& that)
    {
        o3_trace1 trace;

        if (this != &that) {
            tSi<T> tmp = that;

            swap(*this, tmp);
        }
        return *this;
    }

    template<typename T1>
    tSi& operator=(const tSi<T1>& that)
    {
        o3_trace1 trace;

        if (this != (void*) that.ptr()) {
            tSi<T> tmp = that;

            swap(*this, tmp);
        }
        return *this;
    }

    ~tSi()
    {
        o3_trace1 trace;

        if (m_ptr)
            m_ptr->release();
    }

    bool valid() const
    {
        o3_trace1 trace;

        return m_ptr ? true : false;
    }

    T* ptr() const
    {
        o3_trace1 trace;

        return m_ptr;
    }

    operator T*() const
    {
        o3_trace1 trace;

        return ptr();
    }

    T& operator*() const
    {
        o3_trace1 trace;

        return *ptr();
    }

    T* operator->() const
    {
        o3_trace1 trace;

        return ptr();
    }
};

o3_iid(iUnk, 0x00000000,
             0x0000,
             0x0000,
             0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46);

o3_iid(iWeak, 0x43DE1BBC,
              0xD28B,
              0x45AC,
              0xAA, 0x5E, 0x15, 0xBB, 0x93, 0x80, 0xA7, 0x95);

struct iWeak : iUnk {};

}

#endif // O3_I_UNK_H
