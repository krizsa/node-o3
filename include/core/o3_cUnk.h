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
#ifndef O3_C_UNK_H
#define O3_C_UNK_H

#ifdef O3_WIN32
#include <objbase.h>
#endif // O3_WIN32

#define o3_cls(T) \
struct T; \
typedef tSc<T> s##T

#define o3_begin_class(T) \
    virtual int32_t queryInterface(const Guid& iid, void** obj) \
    { \
        o3_trace2 trace; \
        \
        if (T::queryInterface(iid, obj) == 0) \
            return S_OK; \
        else

#define o3_end_class() \
            return E_NOINTERFACE; \
    } \
    \
    virtual uint32_t addRef() \
    { \
        o3_trace2 trace; \
        \
        return cUnk::addRef(); \
    } \
    \
    virtual uint32_t release() \
    { \
        o3_trace2 trace; \
        \
        return cUnk::release(); \
    }

#define o3_add_iface(T) \
    if (iid == iidof((T*) this)) { \
        *obj = (T*) this; \
        addRef(); \
        return S_OK; \
    } else

namespace o3 {

#ifdef O3_POSIX
const int32_t S_OK = 0x00000000;
const int32_t E_NOINTERFACE = 0x80004002;
#endif // O3_POSIX

template<typename T>
class tSc {
    T* m_ptr;

public:
    tSc(T* ptr = 0) : m_ptr(ptr)
    {
        if (m_ptr)
            m_ptr->addRef();
    }

    tSc(const tSc& that) : m_ptr(that.m_ptr)
    {
        o3_trace1 trace;

        if (m_ptr)
            m_ptr->addRef();
    }

    tSc& operator=(const tSc& that)
    {
        o3_trace1 trace;

        if (this != &that) {
            tSc tmp = that;

            swap(*this, tmp);
        }
        return *this;
    }

    ~tSc()
    {
        if (m_ptr)
            m_ptr->release();
    }

    bool valid() const
    {
        o3_trace1 trace;

        return m_ptr;
    }

    T* ptr() const
    {
        o3_trace1 trace;

        return m_ptr;
    }

    operator T*() const
    {
        o3_trace1 trace;

        return m_ptr;
    }

    T& operator*() const
    {
        o3_trace1 trace;

        return m_ptr;
    }

    T* operator->() const
    {
        o3_trace1 trace;

        return m_ptr;
    }

    template<typename T1>
    operator tSi<T1>()
    {
        o3_trace1 trace;

        return m_ptr;
    }
};

o3_cls(cUnk);

struct cUnk : iUnk {
    o3_cls(cWeak);

    struct cWeak : iWeak {
        volatile uint32_t m_ref_count;
        volatile int m_spin_lock;
        iUnk* m_unk_outer;

        cWeak(iUnk* unk_outer) : m_ref_count(0), m_spin_lock(0),
                                 m_unk_outer(unk_outer)
        {
        }

        virtual ~cWeak()
        {
        }

        int32_t queryInterface(const Guid& iid, void** obj)
        {
            o3_trace2 trace;

            if (iid == iidof(this)) {
                *obj = this;
                addRef();
                return S_OK;
            } else {
                int32_t hr;

                lock();
                if (m_unk_outer)
                    hr = m_unk_outer->queryInterface(iid, obj);
                else
                    hr = E_NOINTERFACE;
                unlock();
                return hr;
            }
        }

        uint32_t addRef()
        {
            return atomicInc((volatile int&) m_ref_count);
        }

        uint32_t release()
        {
            if (atomicDec((volatile int&) m_ref_count) == 0) {
                delete this;
                return 0;
            } 
            return m_ref_count;
        }

        void lock()
        {
            o3_trace2 trace;

            while (atomicTas(m_spin_lock))
                ;
        }

        void unlock()
        {
            o3_trace2 trace;

            m_spin_lock = 0;
        }
    };

    volatile uint32_t m_ref_count;
    scWeak m_weak;

#pragma warning(push)
#pragma warning(disable:4355)
    cUnk() : m_ref_count(0), m_weak(new cWeak(this))
    {
        m_weak->m_unk_outer = this;
    }
#pragma warning(pop)

    virtual ~cUnk()
    {
    }

    int32_t queryInterface(const Guid& iid, void** obj)
    {
        o3_trace2 trace;

        if (iid == iidof(this)) {
            *obj = o3_cast this;
            addRef();
        } else if (iid == iidof(m_weak)) {
            *obj = (iWeak*) m_weak;
            m_weak->addRef();
        } else
            return E_NOINTERFACE;
        return S_OK;
    }

    uint32_t addRef()
    {
        o3_trace2 trace;

        return atomicInc((volatile int&) m_ref_count);
    }

    uint32_t release()
    {
        o3_trace2 trace;

        m_weak->lock();
        if (atomicDec((volatile int&) m_ref_count) == 0) {
            m_weak->m_unk_outer = 0;
            m_weak->unlock();
            o3_delete(this);
            return 0;
        }
        m_weak->unlock(); 
        return m_ref_count;
    }
};

}

#endif // O3_C_UNK_H
