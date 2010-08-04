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
#ifndef O3_T_STR_H
#define O3_T_STR_H

namespace o3 {

template<typename C>
class tStr {
    struct Concat {
        Concat(const tStr& left, const tStr& right) : left(left), right(right)
        {
        }

        size_t capacity() const
        {
            return left.capacity() + right.capacity();
        }

        size_t size() const
        {
            return left.size() + right.size();
        }

        void flatten(C** out) const
        {
            left.flatten(out);
            right.flatten(out);
        }

        tStr left;
        tStr right;
    };

    struct Substr {
        Substr(const tStr& str, size_t pos, size_t n) : str(str), pos(pos), n(n)
        {
        }

        size_t capacity() const
        {
            return str.capacity();
        }

        size_t size() const
        {
            return n;
        }

        void flatten(C** out) const
        {
            memCopy(*out, str.ptr() + pos, n * sizeof(C));
            *out += n;
        }

        tStr str;
        size_t pos;
        size_t n;
    };

    union {
        struct {
            iAlloc* alloc;
            union {
                Concat* concat;
                Substr* substr;
            };
        } m_u;
        uint8_t m_buf[sizeof(Buf)];
    };

    tStr(const tStr& left, const tStr& right)
    {
        m_u.alloc = left.alloc();
        m_u.alloc->addRef();
        m_u.concat = o3_new(Concat)(left, right);
        setConcat();
    }

    tStr(const tStr& str, size_t pos, size_t n)
    {
        m_u.alloc = str.alloc();
        m_u.alloc->addRef();
        m_u.substr = o3_new(Substr)(str, pos, n);
        setSubstr();
    }

    bool isConcat() const
    {
        return ((uintptr_t) m_u.alloc & 0x3) == 0x3
            && (uintptr_t) m_u.concat & 0x1;
    }

    bool isSubstr() const
    {
        return ((uintptr_t) m_u.alloc & 0x3) == 0x3
            && (uintptr_t) m_u.substr & 0x2;
    }

    Concat* concat() const
    {
        return (Concat*) ((uintptr_t) m_u.concat & ~(sizeof(int) - 1));
    }    

    Substr* substr() const
    {
        return (Substr*) ((uintptr_t) m_u.substr & ~(sizeof(int) - 1));
    }

    Buf& buf() const
    {
        if (isConcat() || isSubstr()) {
            tStr tmp(size(), alloc());
            C* ptr = tmp.ptr();

            flatten(&ptr);
            tmp.resize(size());
            swap(*(tStr*) this, tmp);
        }
        return *(Buf*) m_buf;
    }

    void flatten(C** out) const
    {
        if (isConcat()) 
            concat()->flatten(out);
        else if (isSubstr()) 
            substr()->flatten(out);
        else {
            memCopy(*out, ptr(), size() * sizeof(C));
            *out += size();
        }
    }

    void setConcat() 
    {
        m_u.alloc = (iAlloc*) ((uintptr_t) m_u.alloc  | 0x3);
        m_u.concat = (Concat*) ((uintptr_t) m_u.concat | 0x1);
    }

    void setSubstr()
    {
        m_u.alloc = (iAlloc*) ((uintptr_t) m_u.alloc  | 0x3);
        m_u.substr = (Substr*) ((uintptr_t) m_u.substr | 0x2);
    }

public:
    static tStr fromBool(bool val, iAlloc* alloc = g_sys)
    {
        o3_trace1 trace;
        const C STR_TRUE[]  = { 't', 'r', 'u', 'e', '\0' };
        const C STR_FALSE[] = { 'f', 'a', 'l', 's', 'e', '\0' };

        return tStr(val ? STR_TRUE : STR_FALSE, alloc);
    }

    static tStr fromInt32(int32_t val, iAlloc* alloc = g_sys)
    {
        o3_trace1 trace;
        const C STR_FORMAT[] = { '%', 'l', 'd', '\0' };
        tStr str(alloc);

        str.appendf(STR_FORMAT, val);
        return str;
    }

    static tStr fromInt64(int64_t val, iAlloc* alloc = g_sys)
    {
        o3_trace1 trace;
        const C STR_FORMAT[] = { '%', 'l', 'l', 'd', '\0' };
        tStr str(alloc);

        str.appendf(STR_FORMAT, val);
        return str;
    }

    static tStr fromDouble(double val, iAlloc* alloc = g_sys)
    {
        o3_trace1 trace;
        const C STR_FORMAT[] = { '%', 'f', '\0' };
        tStr str(alloc);

        str.appendf(STR_FORMAT, val);
        return str;
    }

    static tStr fromHex(const void* ptr, size_t size)
    {
        o3_trace1 trace;
        tStr str(strFromHex((C*) 0, ptr, size));

        str.resize(strFromHex(str.ptr(), ptr, size));
        return str;
    }

    static tStr fromBase64(const void* ptr, size_t size)
    {
        o3_trace1 trace;
        tStr str(strFromBase64((C*) 0, ptr, size));

        str.resize(strFromBase64(str.ptr(), ptr, size));
        return str;
    }

    explicit tStr(size_t capacity, iAlloc* alloc = g_sys)
    {
        o3_trace1 trace;

        new (m_buf) Buf((capacity + 1) * sizeof(C), alloc);
        resize(0);
    }

    explicit tStr(iAlloc* alloc = g_sys)
    {
        o3_trace1 trace;

        new(this) tStr((size_t) 0, alloc);
    }

    tStr(const C* str, size_t len, iAlloc* alloc = g_sys)
    {
        o3_trace1 trace;

        new(m_buf) Buf((len + 1) * sizeof(C), alloc);
        buf().copy(str, len * sizeof(C));
        resize(len);
    }

    tStr(const C* str, iAlloc* alloc = g_sys)
    {
        o3_trace1 trace;

        new(this) tStr(str, str ? strLen(str) : 0, alloc);
    }

    template<typename C1>
    tStr(const C1* str, size_t len, iAlloc* alloc = g_sys)
    {
        o3_trace1 trace;
        tStr tmp(alloc);

        tmp.reserve(strFromStr((C*) 0, str, len));
        tmp.resize(strFromStr(tmp.ptr(), str, len));
        new (this) tStr(tmp);
    }

    template<typename C1>
    tStr(const C1* str, iAlloc* alloc = g_sys)
    {
        o3_trace1 trace;

        new(this) tStr(str, strLen(str), alloc);
    }

    tStr(const Buf& buf)
    {
        size_t size = buf.size() / sizeof(C);

        new (m_buf) Buf(buf);
        resize(size - (((const C*) buf.ptr())[size - 1] == 0));
    }

    tStr(const tStr& that)
    {
        new (m_buf) Buf(that.buf());
    }

    template<typename C1>
    tStr(const tStr<C1>& that)
    {
        o3_trace1 trace;

        new(this) tStr(that.ptr(), that.alloc());
    }

    tStr& operator=(const tStr& that)
    {
        if (this != &that) {
            tStr tmp(that);

            swap(*this, tmp);
        }
        return *this;
    }

    ~tStr()
    {
        if (isConcat()) {
            o3_delete(concat());
            alloc()->release();
        } else if (isSubstr()) {
            o3_delete(substr());
            alloc()->release();
        } else
            ((Buf*) m_buf)->~Buf();
    }

    bool operator==(const tStr& that) const
    {
        o3_trace1 trace;

        return buf() == that.buf();
    }

    bool operator==(const C* str) const
    {
        o3_trace1 trace;

        return strEquals(ptr(), str);
    }

    bool operator<(const tStr& that) const
    {
        o3_trace1 trace;

        return buf() < that.buf();
    }

    bool empty() const
    {
        o3_trace1 trace;

        return size() == 0;
    }

    iAlloc* alloc() const
    {
        o3_trace1 trace;

        return (iAlloc*) ((uintptr_t) m_u.alloc & ~(sizeof(int) - 1));
    }

    size_t capacity() const
    {
        o3_trace1 trace;

        if (isConcat()) 
            return concat()->capacity();
        else if (isSubstr())
            return substr()->capacity();
        else
            return buf().capacity() / sizeof(C) - 1;
    }

    size_t size() const
    {
        o3_trace1 trace;

        if (isConcat())
            return concat()->size();
        else if (isSubstr())
            return substr()->size();
        else
            return buf().size() / sizeof(C) - 1;
    }

    const C* ptr() const
    {
        o3_trace1 trace;

        return (const C*) ((const Buf&) buf()).ptr();
    }

    operator const C*() const
    {
        return ptr();
    }

    void reserve(size_t new_capacity)
    {
        o3_trace1 trace;

        buf().reserve((new_capacity + 1) * sizeof(C));
    }

    void resize(size_t new_size)
    {
        o3_trace1 trace;

        buf().resize((new_size + 1) * sizeof(C));
        if (((const tStr*) this)->ptr()[size()])
            ptr()[size()] = 0;
    }

    C* ptr()
    {
        o3_trace1 trace;

        return (C*) buf().ptr();
    }

    operator C*()
    {
        o3_trace1 trace;

        return ptr();
    }

    bool toBool() const
    {
        o3_trace1 trace;

        return strToBool(ptr());
    }

    int32_t toInt32() const
    {
        o3_trace1 trace;

        return strToInt32(ptr());
    }

    int64_t toInt64() const
    {
        o3_trace1 trace;

        return strToInt64(ptr());
    }

    double toDouble() const
    {
        o3_trace1 trace;

        return strToDouble(ptr());
    }

    Buf toBuf() const
    {
        o3_trace1 trace;

        return buf();
    }

    operator Buf() const
    {
        o3_trace1 trace;

        return toBuf();
    }

    size_t find(size_t pos, const C* str, size_t n) const
    {
        o3_trace1 trace;
        size_t index;

        index = buf().find(pos * sizeof(C), str, n * sizeof(C));
        if (index == NOT_FOUND)
            return NOT_FOUND;
        return index / sizeof(C);
    }

    size_t find(size_t pos, const C* str) const
    {
        o3_trace1 trace;

        return find(pos, str, strLen(str));
    }

    size_t find(const C* str, size_t n) const
    {
        o3_trace1 trace;

        return find(0, str, n);
    }

    size_t find(const C* str) const
    {
        o3_trace1 trace;

        return find(0, str);
    }

	size_t findRight(size_t pos, const C* str, size_t n) const
	{
		o3_trace1 trace;
		size_t index;

		index = buf().findRight(pos * sizeof(C), str, n * sizeof(C));
		if (index == NOT_FOUND)
			return NOT_FOUND;
		return index / sizeof(C);
	}

	size_t findRight(size_t pos, const C* str) const
	{
		o3_trace1 trace;

		return findRight(pos, str, strLen(str));
	}

	size_t findRight(const C* str) const
	{
		o3_trace1 trace;

		return findRight(size()-1, str);
	}


    void insert(size_t pos, C c, size_t n)
    {
        o3_trace1 trace;

        buf().insertPattern(pos * sizeof(C), c, n * sizeof(C));
    }

    void insert(size_t pos, const C* str, size_t n)
    {
        o3_trace1 trace;

        buf().insert(pos * sizeof(C), str, n * sizeof(C));
    }

    void insert(size_t pos, const C* str)
    {
        o3_trace1 trace;

        insert(pos, str, strLen(str));
    }

    void insertfv(size_t pos, const C* format, va_list ap)
    {
        o3_trace1 trace;
        size_t n = strPrintfv((C*) 0, format, ap);
        C* str;
        C  c;

        buf().shift(pos * sizeof(C), n * sizeof(C));
        str = ptr() + pos;
        c = str[n];
        strPrintfv(str, format, ap);
        str[n] = c;    
    }

    void insertf(size_t pos, const C* format, ...)
    {
        o3_trace1 trace;
        va_list ap;

        va_start(ap, format);
        insertfv(pos, format, ap);
        va_end(ap);
    }

    void append(C c, size_t n = 1)
    {
        o3_trace1 trace;

        insert(size(), c, n);
    }

    void append(const C* str, size_t n)
    {
        o3_trace1 trace;

        insert(size(), str, n);
    }

    void append(const C* str)
    {
        o3_trace1 trace;

        insert(size(), str);
    }

    void appendfv(const C* format, va_list ap)
    {
        o3_trace1 trace;

        insertfv(size(), format, ap);
    }

    void appendf(const C* format, ...)
    {
        o3_trace1 trace;
        va_list ap;

        va_start(ap, format);
        appendfv(format, ap);
        va_end(ap);
    }

    tStr& concat(const tStr& right)
    {
        o3_trace1 trace;
        tStr str(*this, right);

        swap(*this, str);
        return *this;
    }

    tStr& operator+=(const tStr& right)
    {
        o3_trace1 trace;

        return concat(right);
    }

    tStr substr(size_t pos, size_t n) const
    {
        o3_trace1 trace;

        return tStr(*this, pos, n);
    }

    tStr substr(size_t pos) const
    {
        o3_trace1 trace;

        return substr(pos, size() - pos);
    }

    void remove(size_t pos, size_t n = 1)
    {
        o3_trace1 trace;

        buf().remove(pos * sizeof(C), n * sizeof(C));
    }

    void clear()
    {
        o3_trace1 trace;

        remove(0, size());
    }

    void replace(size_t pos, size_t n, C c, size_t n1 = 1)
    {
        o3_trace1 trace;

        buf().replace(pos * sizeof(C), n * sizeof(C), c, n1 * sizeof(C));
    }

    void replace(size_t pos, size_t n, const C* str, size_t n1)
    {
        o3_trace1 trace;

        buf().replace(pos * sizeof(C), n * sizeof(C), str, n1 * sizeof(C));
    }

    void replace(size_t pos, size_t n, const C* str)
    {
        o3_trace1 trace;

        replace(pos, n, str, strLen(str));
    }

    void replacefv(size_t pos, size_t n, const C* format, va_list ap)
    {
        o3_trace1 trace;
        size_t n1 = strPrintf((C*) 0, format, ap); 
        const C* str;
        C c;

        if (n < n1)
            buf().shift(pos * sizeof(C), (n1 - n) * sizeof(C));
        else if (n > n1)
            remove(pos, n - n1);
        str = ptr() + pos;
        c = str[n];
        strPrintfv(str, format, ap);
        str[n] = c;    
    }

    void replacef(size_t pos, size_t n, const C* format, ...)
    {
        o3_trace1 trace;
        va_list ap;

        va_start(ap, format);
        replacefv(pos, n, format, ap);
        va_end(ap);
    }

	void findAndReplaceAll(const C* orig, const C* to)
	{
		buf().findAndReplaceAll(orig,strLen(orig)*sizeof(C), to, strLen(to)*sizeof(C));
	}
};

template<typename C, typename C1>
inline tStr<C> operator+(const C* str1, const tStr<C1>& str2)
{
    o3_trace1 trace;
    tStr<C> tmp = str1;

    return tmp += str2;
}

template<typename C, typename C1>
inline tStr<C> operator+(const tStr<C>& str1, const C1* str2)
{
    o3_trace1 trace;
    tStr<C> tmp = str1;

    return tmp += str2;
}

template<typename C, typename C1>
inline tStr<C> operator+(const tStr<C>& str1, const tStr<C1>& str2)
{
    o3_trace1 trace;
    tStr<C> tmp = str1;

    return tmp += str2;
}

}

#endif // O3_T_STR_H
