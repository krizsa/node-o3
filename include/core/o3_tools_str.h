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
#ifndef O3_TOOLS_STR_H
#define O3_TOOLS_STR_H

#include <string.h>
#include <wchar.h>
#include <stdlib.h>
#include <stdio.h>

namespace o3 {

inline size_t strLen(const char* str)
{
    o3_trace0 trace;

    return ::strlen(str);
}

inline size_t strLen(const wchar_t* str)
{
    o3_trace0 trace;

    return ::wcslen(str);
}

inline int strCompare(const char* str, const char* str1, size_t n)
{
    o3_trace0 trace;

    return ::strncmp(str, str1, n);
}

inline int strCompare(const wchar_t* str, const wchar_t* str1, size_t n)
{
    o3_trace0 trace;

    return ::wcsncmp(str, str1, n);
}

template<typename C>
inline int strCompare(const C* str, const C* str1)
{
    o3_trace0 trace;
    size_t n1 = strLen(str);
    size_t n2 = strLen(str1);
    return strCompare(str, str1, max(n1, n2));
}

template<typename C>
inline bool strEquals(const C* str, const C* str1)
{
    o3_trace0 trace;

    return strCompare(str, str1) == 0;
}

template<typename C>
inline bool strEquals(const C* str, const C* str1, size_t n)
{
    o3_trace0 trace;

    return strCompare(str, str1, n) == 0;
}

inline int strCaseCompare(const char* str, const char* str1, size_t n)
{
	o3_trace0 trace;
#ifdef O3_WIN32
	return ::_strnicmp(str, str1, n);
#else
	return ::strncasecmp(str, str1, n);
#endif
}

inline int strCaseCompare(const wchar_t* str, const wchar_t* str1, size_t n)
{
	o3_trace0 trace;
#ifdef O3_WIN32
	return ::_wcsnicmp(str, str1, n);
#else
	return ::wcsncasecmp(str, str1, n);
#endif

}

template<typename C>
inline int strCaseCompare(const C* str, const C* str1)
{
	o3_trace0 trace;
	size_t n1 = strLen(str);
	size_t n2 = strLen(str1);
	return strCaseCompare(str, str1, max(n1, n2));
}

template<typename C>
inline bool strCaseEquals(const C* str, const C* str1)
{
	o3_trace0 trace;

	return strCaseCompare(str, str1) == 0;
}

template<typename C>
inline bool strCaseEquals(const C* str, const C* str1, size_t n)
{
	o3_trace0 trace;

	return strCaseCompare(str, str1, n) == 0;
}

inline char* strCopy(char* dst, const char* src)
{
	return ::strcpy(dst, src);
}
	
inline wchar_t* strCopy(wchar_t* dst, const wchar_t* src)
{
	return ::wcscpy(dst, src);
}

template<typename C>
inline bool strToBool(const C* str)
{
    o3_trace0 trace;
    const C STR_TRUE[] = { 't', 'r', 'u', 'e', '\0' };

    return strEquals(str, STR_TRUE) || strToInt32(str);
}

inline int32_t strToInt32(const char* str)
{
    o3_trace0 trace;

    return ::strtol(str, 0, 0);
}

inline int32_t strToInt32(const wchar_t* str)
{
    o3_trace0 trace;

    return ::wcstol(str, 0, 0);
}

inline int64_t strToInt64(const char* str)
{
    o3_trace0 trace;

#ifdef O3_WIN32
    return ::_strtoi64(str, 0, 0);
#else
    return ::strtoll(str, 0, 0);
#endif
}

inline int64_t strToInt64(const wchar_t* str)
{
    o3_trace0 trace;

#ifdef O3_WIN32
    return ::_wcstoi64(str, 0, 0);
#else
    return ::wcstoll(str, 0, 0);
#endif
}

inline double strToDouble(const char* str)
{
    o3_trace0 trace;

    return ::strtod(str, 0);
}

inline double strToDouble(const wchar_t* str)
{
    o3_trace0 trace;

    return ::wcstod(str, 0);
}

inline size_t strPrintfv(char* str, const char* format, va_list ap)
{
    o3_trace0 trace;

    return ::vsnprintf(str, str ? (size_t) -1 : 0, format, ap);
}

inline size_t strPrintfv(wchar_t* str, const wchar_t* format, va_list ap) 
{
    o3_trace0 trace;

#ifdef O3_WIN32
    return ::vswprintf(str,  str ? (size_t) -1 : 0, format, ap);
#else
	return ::vswprintf(str, (size_t) -1, format, ap);
#endif
}

template<typename C>
inline size_t strFromHex(C* str, const void* ptr, size_t size)
{
    o3_trace0 trace;
    uint8_t* ptr1 = (uint8_t*) ptr;
    size_t len = 0;

    while (size--) {
        unsigned bits = *ptr1++;

        if (str) {
            *str++ = chrFromHex(bits >> 4);
            *str++ = chrFromHex(bits & 0xF);
        }
        len += 2;
        if (size != 0) {
            if (str)
                *str++ = ' ';
            ++len;
        }
    }
    if (str)
        *str = '\0';
    return len;
}

template<typename C>
inline size_t strFromBase64(C* str, const void* ptr, size_t size)
{
    o3_trace0 trace;
    uint8_t* ptr1 = (uint8_t*) ptr;
    size_t len = 0;
    unsigned bits = 0;
    int n = 0;
    int cols = 0;

    while (size--) {
        bits |= *ptr1++;
        if (++n == 3) {
            if (str) {
                *str++ = chrFromBase64(bits >> 18);
                *str++ = chrFromBase64(bits >> 12 & 0x3F);
                *str++ = chrFromBase64(bits >> 6 & 0x3F);
                *str++ = chrFromBase64(bits & 0x3F);
            }
            len += 4, cols += 4;
            if (cols == 72) {
                if (str)
                    *str++ = '\n';
                ++len;
                cols = 0;
            }
            bits = n = 0;
        } else
            bits <<= 8; 
    }
    if (n > 0) {
        bits <<= 16 - 8 * n;
        if (str) {
            *str++ = chrFromBase64(bits >> 18);
            *str++ = chrFromBase64(bits >> 12 & 0x3F);
            *str++ = n > 1 ? chrFromBase64(bits >> 6 & 0x3F) : '=';
            *str++ = '=';
        }
        len += 4;
    }
    if (str)
        *str = '\0';
    return len;
}

inline size_t strFromStr(char* str1, const wchar_t* str, size_t len)
{
    o3_trace0 trace;
    size_t len1 = 0;

    while (len--) {
        unsigned code;

        if (chrIsBetween((wchar_t) 0xD800, *str, (wchar_t) 0xDBFF)) {
            code = (*str++ - 0xD800) << 10;
            if (!chrIsBetween((wchar_t) 0xDC00, *str, (wchar_t) 0xDFFF))
                goto error;
            code |= *str++ - 0xDC00;
        } else if ((code = *str) != 0)
            ++str;
        if (code < 0x80) {
            if (str1)
                *str1++ = (char) code;
            ++len1; 
        } else if (code < 0x800) {
            if (str1) {
                *str1++ = (char) (0xC0 | code >> 6);
                *str1++ = (char) (0x80 | (code & 0x3F));
            }
            len1 += 2;
        } else if (code < 0x10000) {
            if (str1) {
                *str1++ = (char) (0xE0 | code >> 12);
                *str1++ = (char) (0x80 | (code >> 6 & 0x3F));
                *str1++ = (char) (0x80 | (code & 0x3F));
            }
            len1 += 3;
        } else if (code < 0x11000) {
            if (str1) {
                *str1++ = (char) (0xF0 | code >> 18);
                *str1++ = (char) (0x80 | (code >> 12 & 0x3F));
                *str1++ = (char) (0x80 | (code >> 6 & 0x3F));
                *str1++ = (char) (0x80 | (code & 0x3F));
            }
            len1 += 4;
        }
    }
error:
    if (str1)
        *str1 = 0;
    return len1;
}

inline size_t strFromStr(wchar_t* str1, const char* str, size_t len)
{
    o3_trace0 trace;
    size_t len1 = 0;

    while (len--) {
        unsigned code;

        if (*str & 0x80) {
            char c = *str++;
            int seq = 0;

            while (c << seq & 0x80)
                ++seq;
            if (seq < 2 || seq > 4)
                goto error;
            code = c & ((1 << (7 - seq)) - 1);
            while (--seq && (c = *str++) && !((c ^ 0x80) & 0xC0))
                code = code << 6 | (c & 0x3F);
            if (seq > 0)
                goto error;
        } else if ((code = *str) != 0)
            ++str;
        if (code < 0x10000) {
            if (str1)
                *str1++ = (wchar_t) code;
            ++len1;
        } else {
            code -= 0x10000;
            if (str1) {
                *str1++ = (wchar_t) (0xD800 + (code >> 10));
                *str1++ = (wchar_t) (0xDC00 + (code & 0x3F));
            }
            len1 += 2;
        }
    }
error:
    if (str1)
        *str1 = 0;
    return len1;
}

}

#endif // O3_TOOLS_STR_H
