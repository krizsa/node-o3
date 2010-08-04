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
#ifndef O3_VAR_H
#define O3_VAR_H

namespace o3 {

class Var {
    union {
        struct {
            iAlloc* alloc;
            int type;
            union {
                bool val_bool;
                int32_t val_int32;
                int64_t val_int64;
                double val_double;
                iScr* val_scr;
            };
        } m_u;    
        uint8_t m_str[sizeof(Str)];
        uint8_t m_wstr[sizeof(Str)];
    };

    bool isStr() const
    {
        o3_trace1 trace;

        return ((uintptr_t) m_u.alloc & 0x3) ? true : false;
    }

    bool isWide() const
    {
        o3_trace1 trace;

        return !((uintptr_t) m_u.alloc & 0x1) &&
               ((uint8_t*) this)[O3_AUTO_CAPACITY + 1] & 0x2;
    }

    void setStr()
    {
        o3_trace1 trace;

        if (!((uintptr_t) m_u.alloc & 0x1))
            m_u.alloc = (iAlloc*) ((uintptr_t) m_u.alloc | 0x2);
    }

    void setWide()
    {
        o3_trace1 trace;

        ((uint8_t*) this)[O3_AUTO_CAPACITY + 1] |= 0x2;
    }

public:
    enum Type {
        TYPE_VOID,
        TYPE_NULL,
        TYPE_BOOL,
        TYPE_INT32,
        TYPE_INT64,
        TYPE_DOUBLE,        
        TYPE_STR,
        TYPE_WSTR,
		TYPE_SCR
    };

    Var(iAlloc* alloc = g_sys)
    {
        o3_trace1 trace;

        m_u.alloc = alloc;
        m_u.alloc->addRef();
        m_u.type = TYPE_VOID;
    }

    Var(bool val, iAlloc* alloc = g_sys)
    {
        o3_trace1 trace;

        m_u.alloc = alloc;
        m_u.alloc->addRef();
        m_u.type = TYPE_BOOL;
        m_u.val_bool = val;
    }

    Var(int32_t val, iAlloc* alloc = g_sys)
    {
        o3_trace1 trace;

        m_u.alloc = alloc;
        m_u.alloc->addRef();
        m_u.type = TYPE_INT32;
        m_u.val_int32 = val;
    }

    Var(int64_t val, iAlloc* alloc = g_sys)
    {
        o3_trace1 trace;

        m_u.alloc = alloc;
        m_u.alloc->addRef();
        m_u.type = TYPE_INT64;
        m_u.val_int64 = val;
    }

    Var(size_t val, iAlloc* alloc = g_sys)
    {
        o3_trace1 trace;

        new (this) Var((int32_t) val, alloc);
    }
 
    Var(double val, iAlloc* alloc = g_sys)
    {
        o3_trace1 trace;

        m_u.alloc = alloc;
        m_u.alloc->addRef();
        m_u.type = TYPE_DOUBLE;
        m_u.val_double = val;
    }

    Var(iScr* val, iAlloc* alloc = g_sys)
    {
        o3_trace1 trace;
        
        m_u.alloc = alloc;
        m_u.alloc->addRef();
        if (val) {
            m_u.type = TYPE_SCR;
            m_u.val_scr = val;
            m_u.val_scr->addRef();
        } else 
            m_u.type = TYPE_NULL;
    }

    template<typename T>
    Var(const tSi<T>& val, iAlloc* alloc = g_sys)
    {
        o3_trace1 trace;

        new(this) Var(siScr(val).ptr(), alloc);
    }

    Var(const Str& str)
    {
        o3_trace1 trace;

        new(m_str) Str(str);
        setStr();
    }

    Var(const char* str, iAlloc* alloc = g_sys)
    {
        o3_trace1 trace;

        new(this) Var(Str(str, alloc));
    }

    Var(const WStr& str)
    {
        o3_trace1 trace;

        new(m_wstr) WStr(str);
        ((WStr*) m_wstr)->reserve(O3_AUTO_CAPACITY);
        setStr();
        setWide();
    }

    Var(const wchar_t* str, iAlloc* alloc = g_sys)
    {
        o3_trace1 trace;

        new(this) Var(WStr(str, alloc));
    }

    Var(const Var& that)
    {
        o3_trace1 trace;

        if (that.isStr()) {
            if (that.isWide()) {
                new(m_wstr) WStr(*(WStr*) that.m_wstr);
                setWide();
            } else
                new(m_str) Str(*(Str*) that.m_str);
            setStr();
        } else {
            memCopy(this, &that, sizeof(Var));
            alloc()->addRef();
            if (m_u.type == TYPE_SCR)
                m_u.val_scr->addRef();
        }
    }

    Var& operator=(const Var& that)
    {
        o3_trace1 trace;

        switch (that.type()) {
        case TYPE_VOID:
            return set();
        case TYPE_NULL:
            return set((iScr*) 0);
        case TYPE_BOOL:
            return set(that.toBool());
        case TYPE_INT32:
            return set(that.toInt32());
        case TYPE_INT64:
            return set(that.toInt64());
        case TYPE_DOUBLE:
            return set(that.toDouble());
        case TYPE_SCR:
            return set(that.toScr());
        case TYPE_STR:
            return set(that.toStr());
        case TYPE_WSTR:
            return set(that.toWStr());
        default:
            return *this;
        }
    }

    ~Var()
    {
        o3_trace1 trace;

        if (isStr()) {
            if (isWide())
                ((WStr*) m_wstr)->~WStr();
            else
                ((Str*) m_str)->~Str();
        } else {
            if (m_u.type == TYPE_SCR)
                m_u.val_scr->release();
            alloc()->release();
        }
    }

    Type type() const
    {
        o3_trace1 trace;

        return isStr() ? (isWide() ? TYPE_WSTR : TYPE_STR) : (Type) m_u.type;
    }

    iAlloc* alloc() const
    {
        o3_trace1 trace;

        return (iAlloc*) ((uintptr_t) m_u.alloc & ~(sizeof(int) - 1));
    }

    bool toBool() const
    {
        o3_trace1 trace;

        switch (type()) {
        case TYPE_BOOL:
            return m_u.val_bool;
        case TYPE_INT32:
            return m_u.val_int32 ? true : false;
        case TYPE_INT64:
            return m_u.val_int64 ? true : false;
        case TYPE_DOUBLE:
            return m_u.val_double ? true : false;
        case TYPE_SCR:
            return m_u.val_scr ? true : false;
        case TYPE_WSTR:
            return ((WStr*) m_wstr)->toBool();
        case TYPE_STR:
            return ((Str*) m_str)->toBool();
        default:
            return 0;
        }
    }

    int32_t toInt32() const
    {
        o3_trace1 trace;

        switch (type()) {
        case TYPE_BOOL:
            return m_u.val_bool;
        case TYPE_INT32:
            return m_u.val_int32;
        case TYPE_INT64:
            return (int32_t)m_u.val_int64;
        case TYPE_DOUBLE:
            return (int)DoubleToInt(m_u.val_double);
        case TYPE_WSTR:
            return ((WStr*) m_wstr)->toInt32();
        case TYPE_STR:
            return ((Str*) m_str)->toInt32();
        default:
            return 0;
        }
    }

    int64_t toInt64() const
    {
        o3_trace1 trace;

        switch (type()) {
        case TYPE_BOOL:
            return m_u.val_bool;
        case TYPE_INT32:
            return m_u.val_int32;
        case TYPE_INT64:
            return m_u.val_int64;
        case TYPE_DOUBLE:
            return (int)DoubleToInt(m_u.val_double);
        case TYPE_WSTR:
            return ((WStr*) m_wstr)->toInt64();
        case TYPE_STR:
            return ((Str*) m_str)->toInt64();
        default:
            return 0;
        }
    }

    double toDouble() const
    {
        o3_trace1 trace;

        switch (type()) {
        case TYPE_BOOL:
            return m_u.val_bool;
        case TYPE_INT32:
            return m_u.val_int32;
        case TYPE_INT64:
            return (double)m_u.val_int64;
        case TYPE_DOUBLE:
            return m_u.val_double;
        case TYPE_WSTR:
            return ((WStr*) m_wstr)->toDouble();
        case TYPE_STR:
            return ((Str*) m_str)->toDouble();
        default:
            return 0;
        }
    }

    siScr toScr() const
    {
        o3_trace1 trace;

        switch (type()) {
        case TYPE_SCR:
            return m_u.val_scr;
        default:
            return siScr();
        };
    }

    Str toStr() const
    {
        o3_trace1 trace;

        switch (type()) {
        case TYPE_VOID:
            return "undefined";
        case TYPE_NULL:
            return "null";
        case TYPE_BOOL:
            return Str::fromBool(m_u.val_bool);
        case TYPE_INT32:
            return Str::fromInt32(m_u.val_int32);
        case TYPE_INT64:
            return Str::fromInt64(m_u.val_int64);
        case TYPE_DOUBLE:
            return Str::fromDouble(m_u.val_double);
        case TYPE_SCR:
            return "object";
        case TYPE_WSTR:
            return *(WStr*) m_wstr;
        case TYPE_STR:
            return *(Str*) m_str;
        default:
            return Str();
        }
    }

    WStr toWStr() const
    {
        o3_trace1 trace;

        switch (type()) {
        case TYPE_VOID:
            return L"undefined";
        case TYPE_NULL:
            return L"null";
        case TYPE_BOOL:
            return WStr::fromBool(m_u.val_bool);
        case TYPE_INT32:
            return WStr::fromInt32(m_u.val_int32);
        case TYPE_INT64:
            return WStr::fromInt64(m_u.val_int64);
        case TYPE_DOUBLE:
            return WStr::fromDouble(m_u.val_double);
        case TYPE_SCR:
            return L"object"; 
        case TYPE_WSTR:
            return *(WStr*) m_wstr;
        case TYPE_STR:
            return *(Str*) m_str;
        default:
            return WStr();
        }
    }

    Var& set()
    {
        o3_trace1 trace;
        Var tmp(alloc());

        swap(*this, tmp);
        return *this;
    }

    Var& set(bool val)
    {
        o3_trace1 trace;
        Var tmp(val, alloc());

        swap(*this, tmp);
        return *this;
    }

    Var& set(int32_t val)
    {
        o3_trace1 trace;
        Var tmp(val, alloc());

        swap(*this, tmp);
        return *this;
    }

    Var& set(int64_t val)
    {
        o3_trace1 trace;
        Var tmp(val, alloc());

        swap(*this, tmp);
        return *this;
    }

    Var& set(double val)
    {
        o3_trace1 trace;
        Var tmp(val, alloc());

        swap(*this, tmp);
        return *this;
    }

    Var& set(iScr* scr)
    {
        o3_trace1 trace;
        Var tmp(scr, alloc());

        swap(*this, tmp);
        return *this;
    }

    Var& set(siScr scr)
    {
        o3_trace1 trace;

        return set(scr.ptr());
    }

    Var& set(const Str& str)
    {
        o3_trace1 trace;
        Var tmp = alloc() == str.alloc() ? Var(str) : Var(str, alloc());

        swap(*this, tmp);
        return *this;
    }

    Var& set(const char* str)
    {
        o3_trace1 trace;
        Var tmp(str, alloc());

        swap(*this, tmp);
        return *this;
    }

    Var& set(const WStr& str)
    {
        o3_trace1 trace;
        Var tmp = alloc() == str.alloc() ? Var(str) : Var(str, alloc());

        swap(*this, tmp);
        return *this;
    }

    Var& set(const wchar_t* str)
    {
        o3_trace1 trace;
        Var tmp(str, alloc());

        swap(*this, tmp);
        return *this;
    }
};

}

#endif // O3_VAR_H
