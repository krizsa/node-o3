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
#ifndef O3_BUF_H
#define O3_BUF_H

namespace o3 {

const size_t NOT_FOUND = (size_t) -1;

class Buf {
    iAlloc* m_alloc;
    union {
        struct {
            char size;
            uint8_t data[O3_AUTO_CAPACITY];
        } m_u0;
        struct {
            size_t capacity;
            size_t size;
            void* ptr;
        } m_u1;
    };

    bool isAuto() const
    {
        return (uintptr_t) m_alloc & 0x1;
    }

    bool isWrap() const
    {
        return (uintptr_t) m_u1.ptr & 0x1;
    }

    unsigned& refCount() const
    {
        return *(unsigned*) ((uint8_t*) m_u1.ptr + m_u1.capacity);
    }

    void setAuto()
    {
        m_alloc = (iAlloc*) ((uintptr_t) m_alloc | 0x1);
    }

    void setWrap()
    {
        m_u1.ptr = (void*) ((uintptr_t) m_u1.ptr | 0x1);
    }

    friend void test_Buf();

public:
    template<typename C>
    static Buf fromHex(const C* str, iAlloc* alloc = g_sys)
    {
        o3_trace1 trace;
        Buf buf(memFromHex(0, str), alloc);

        buf.resize(memFromHex(buf.ptr(), str));
        return buf;
    }

    template<typename C>
    static Buf fromBase64(const C* str, iAlloc* alloc = g_sys)
    {
        o3_trace1 trace;
        Buf buf(memFromBase64(0, str), alloc);

        buf.resize(memFromBase64(buf.ptr(), str));
        return buf;
    }

    explicit Buf(size_t capacity, iAlloc* alloc = g_sys) : m_alloc(alloc)
    {
        o3_trace1 trace;

        m_alloc->addRef();
        if (capacity <= (size_t) O3_AUTO_CAPACITY) {
            setAuto();
            m_u0.size = 0;
        } else {
            m_u1.capacity = 1;
            while (m_u1.capacity < capacity)
                m_u1.capacity <<= 1;
            m_u1.size = 0;
            m_u1.ptr = m_alloc->alloc(m_u1.capacity + sizeof(unsigned));
            refCount() = 1;
        }
    }

    explicit Buf(iAlloc* alloc = g_sys)
    {
        o3_trace1 trace;

        new(this) Buf((size_t) 0, alloc);
    }

    Buf(void* ptr, size_t size, iAlloc* alloc = g_sys) : m_alloc(alloc)
    {
        o3_trace1 trace;

        m_alloc->addRef();
        m_u1.capacity = size;
        m_u1.size = size;
        m_u1.ptr = ptr;
        setWrap();
    }

    Buf(iBuf* buf)
    {
        o3_trace1 trace;

        new(this) Buf(buf->unwrap());
    }

    Buf(iStream* stream, iAlloc* alloc = g_sys)
    {
        o3_trace1 trace;
        const size_t SIZE = 1024;
        uint8_t data[SIZE];
        size_t size;

        new(this) Buf(alloc);
        do {
            size = stream->read(data, SIZE);
            append(data, size);
        } while (size == SIZE);
    }

    Buf(iStream* stream, size_t n, iAlloc* alloc = g_sys)
    {
        o3_trace1 trace;

        new(this) Buf(n, alloc);
        stream->read(ptr(), n);
        resize(n);
    }

    Buf(const Buf& that)
    {
        o3_trace1 trace;

        memCopy(this, &that, sizeof(Buf));
        alloc()->addRef();
        if (!isAuto() && !isWrap()) 
            atomicInc((volatile int&) refCount());
    }

    Buf& operator=(const Buf& that)
    {
        o3_trace1 trace;

        if (this != &that) {
            Buf tmp = that;

            swap(*this, tmp);
        }
        return *this;
    }

    ~Buf()
    {
        o3_trace1 trace;

        if (!isAuto() && !isWrap() &&
            atomicDec((volatile int&) refCount()) == 0)
            alloc()->free(ptr());
        alloc()->release();
    }

    bool operator==(const Buf& that) const
    {
        o3_trace1 trace;

        return this == &that
            || (size() == that.size()
                && memEquals(ptr(), that.ptr(), that.size()));
    }

    bool operator<(const Buf& that) const
    {
        o3_trace1   trace;
        int         cmp = memCompare(ptr(),
                                     that.ptr(),
                                     max(size(), that.size()));

        return cmp < 0 ? true : cmp > 0 ? false : size() < that.size();
    }

    bool empty() const
    {
        o3_trace1 trace;

        return size() == 0;
    }

    bool unique() const
    {
        o3_trace1 trace;

        return isAuto() || isWrap() || refCount() == 1;
    }

    iAlloc* alloc() const
    {
        o3_trace1 trace;

        return (iAlloc*) ((uintptr_t) m_alloc & ~(sizeof(int) - 1));
    }

    size_t capacity() const
    {
        o3_trace1 trace;

        return isAuto() ? O3_AUTO_CAPACITY : m_u1.capacity;
    }

    size_t size() const
    {
        o3_trace1 trace;

        return isAuto() ? m_u0.size : m_u1.size;
    }

    const void* ptr() const
    {
        o3_trace1 trace;

        if (isAuto()) 
            return m_u0.data;
        else
            return (const void*) ((uintptr_t) m_u1.ptr & ~(sizeof(int) - 1));
    }

    size_t find(size_t pos, const void* ptr, size_t n) const
    {
        o3_trace1 trace;
        uint8_t* ptr1 = (uint8_t*) this->ptr();
        uint8_t* ptr2 = (uint8_t*) memFind(ptr1 + pos, ptr, size() - pos, n);

        return ptr2 ? ptr2 - ptr1 : NOT_FOUND;
    }

    size_t find(const void* ptr, size_t n) const
    {
        o3_trace1 trace;

        return find(0, ptr, n);
    }

	size_t findRight(size_t pos, const void* ptr, size_t n) const
	{
		o3_trace1 trace;
		uint8_t* ptr1 = (uint8_t*) this->ptr();
		uint8_t* ptr2 = (uint8_t*) memFindReverse(ptr1 + pos, ptr, pos, n);

		return ptr2 ? ptr2 - ptr1 : NOT_FOUND;
	}

	size_t findRight(const void* ptr, size_t n) const
	{
		o3_trace1 trace;

		return findRight(0, ptr, n);
	}

    void reserve(size_t new_capacity)
    {
        o3_trace1 trace;

        if (new_capacity > capacity()) {
            Buf tmp(new_capacity, alloc());

            tmp.append(((const Buf*) this)->ptr(), size());
            swap(*this, tmp);
        }
    }

    void resize(size_t new_size)
    {
        o3_trace1 trace;

        if (new_size > capacity())
            reserve(new_size);
        if (isAuto())
            m_u0.size = (char) new_size;
        else
            m_u1.size = new_size;
    }

    void* ptr() 
    {
        o3_trace1 trace;

        if (!isAuto() && !isWrap() && refCount() > 1) {
            Buf tmp(capacity(), alloc());

            tmp.append(((const Buf*) this)->ptr(), size());
            swap(*this, tmp);
        }
        return (void*) ((const Buf*) this)->ptr();
    }

    void* detach()
    {
        void* ptr = this->ptr();

        setWrap();
        return ptr;
    }

    template<typename T>
    void* set(size_t pos, const T& x, size_t n)
    {
        o3_trace1 trace;

        return memSet((uint8_t*) this->ptr() + pos, x, n);
    }

    void* copy(size_t pos, const void* ptr, size_t n)
    {
        o3_trace1 trace;

        return memCopy((uint8_t*) this->ptr() + pos, ptr, n);
    }

    void* copy(const void* ptr, size_t n)
    {
        o3_trace1 trace;

        return copy(0, ptr, n);
    }

    void* move(size_t pos, size_t pos1, size_t n)
    {
        o3_trace1 trace;

        return memMove((uint8_t*) ptr() + pos, (uint8_t*) ptr() + pos1, n);
    }

    void shift(size_t pos, size_t n)
    {
        o3_trace1 trace;
        size_t new_size = size() + n;
        size_t pos1 = pos + n;

        n = size() - pos;
        if (new_size > capacity()) {
            Buf tmp(new_size, alloc());

            tmp.copy(ptr(), pos);
            tmp.copy(pos1, (uint8_t*) ptr() + pos, n);
            swap(*this, tmp);
        } else
            move(pos1, pos, n);
        resize(new_size);
    }

    template<typename T>
    void insertPattern(size_t pos, const T& x, size_t n)
    {
        o3_trace1 trace;

        shift(pos, n);
        set(pos, x, n);
    }

    void insert(size_t pos, const void* ptr, size_t n)
    {
        o3_trace1 trace;

        shift(pos, n);
        copy(pos, ptr, n);
    }

    template<typename T>
    void appendPattern(const T& x, size_t n)
    {
        o3_trace1 trace;

        insertPattern(size(), x, n);
    }

    void append(const void* ptr, size_t n)
    {
        o3_trace1 trace;

        insert(size(), ptr, n);
    }

    void remove(size_t pos, size_t n = 1)
    {
        o3_trace1 trace;
        size_t pos1 = pos + n;

        move(pos, pos1, size() - pos1);
        resize(size() - n);
    }

    void clear()
    {
        o3_trace1 trace;

        remove(0, size());
    }

    template<typename T>
    void replace(size_t pos, size_t n, const T& x, size_t n1 = 1)
    {
        o3_trace1 trace;

        if (n < n1)
            shift(pos, n1 - n);
        else if (n > n1)
            remove(pos, n - n1);
        set(pos, x, n1);
    }

    void replace(size_t pos, size_t n, const void* ptr, size_t n1)
    {
        o3_trace1 trace;

        if (n < n1)
            shift(pos, n1 - n);
        else if (n > n1)
            remove(pos, n - n1);
        copy(pos, ptr, n1);
    }

    template<typename T>
    void replace(size_t n, const T& x, size_t n1 = 1)
    {
        o3_trace1 trace;

        replace(0, n, x, n1);
    }

    void replace(size_t n, const void* ptr, size_t n1)
    {
        o3_trace1 trace;

        replace(0, n, ptr, n1);
    }

    void findAndReplaceAll(const void* from, size_t n1, const void* to, size_t n2)
    {
        o3_trace1 trace;

        size_t next_match = 0;
        while ( (next_match = find(next_match, from, n1)) != NOT_FOUND) {
            replace(next_match, n1, to, n2);
        }
    }
};

}

#endif // O3_BUF_H
