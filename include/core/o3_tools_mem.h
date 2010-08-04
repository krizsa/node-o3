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
#ifndef O3_TOOLS_MEM_H
#define O3_TOOLS_MEM_H

#include <stdio.h>
#include <new>
#include <string.h>

#define o3_new(T) new(o3::memAlloc(sizeof(T))) T

namespace o3 {

template<typename T>
inline void o3_delete(T* ptr)
{
    ptr->~T();
    memFree(ptr);
}

inline void* memAlloc(size_t size);

inline void memFree(void* ptr);

inline int memCompare(const void* ptr, const void* ptr1, size_t n)
{
    o3_trace0 trace;

    return ::memcmp(ptr, ptr1, n);
} 

inline bool memEquals(const void* ptr, const void* ptr1, size_t n)
{
    o3_trace0 trace;

    return memCompare(ptr, ptr1, n) ? false : true;
}

inline void* memFind(const void* ptr, const void* ptr1, size_t n, size_t n1)
{
    o3_trace0 trace;
    uint8_t* ptr2 = (uint8_t*) ptr;

    for (; n >= n1; ++ptr2, --n)
        if (memEquals(ptr2, ptr1, n1))
            return ptr2;
    return 0;
}

inline void* memFindReverse(const void* ptr, const void* ptr1, size_t n, size_t n1)
{
	o3_trace0 trace;
	uint8_t* ptr2 = (uint8_t*) ptr;	
	if(n1>n)
		return 0;

	for (size_t n2 = n-n1; n2 != 0; --ptr2, --n2)
		if (memEquals(ptr2, ptr1, n1))
			return ptr2;
	return 0;
}

inline void* memSet(void* dst, uint8_t b, size_t n)
{
    o3_trace0 trace;

    return ::memset(dst, b, n);
}

template<typename T>
inline void* memSet(void* dst, const T& x, size_t n)
{
    o3_trace0 trace;
    T* dst1 = (T*) dst;

    for (; n > sizeof(T); n -= sizeof(T)) 
        memCopy((void*) dst1++, &x, sizeof(T));
    memCopy((void*) dst1, &x, n);
    return dst;
}

inline void* memCopy(void* dst, const void* src, size_t n)
{
    o3_trace0 trace;

    return ::memcpy(dst, src, n);
}

inline void* memMove(void* dst, const void* src, size_t n)
{
    o3_trace0 trace;

    return ::memmove(dst, src, n);
}

template<typename C>
inline size_t memFromHex(void* ptr, const C* str)
{
    o3_trace0 trace;
    uint8_t* ptr1 = (uint8_t*) ptr;
    size_t size = 0;
    unsigned bits = 0;
    int n = 0;

    while (C c = *str++) {
        int x;

        if (chrIsSpace(c))
            continue;
        x = chrToHex(c);
        if (x < 0)
            goto error;
        bits |= x;
        if (++n == 2) {
            if (ptr1)
                *ptr1++ = (uint8_t) bits;
            ++size;
            bits = n = 0;
        } else
            bits <<= 4;
    }
error:
    return size;
}

template<typename C>
inline size_t memFromBase64(void* ptr, const C* str)
{
    o3_trace0 trace;
    uint8_t* ptr1 = (uint8_t*) ptr;
    size_t size = 0;
    unsigned bits = 0;
    int n = 0;
    C c;

    while (c = *str++) {
        int x;

        if (c == '=')
            break;
        if (chrIsSpace(c))
            continue;
        x = chrToBase64(c);
        if (x < 0)
            goto error;
        bits |= x;
        if (++n == 4) {
            if (ptr1) {
                *ptr1++ = (uint8_t) (bits >> 16);
                *ptr1++ = (uint8_t) (bits >> 8 & 0xFF);
                *ptr1++ = (uint8_t) (bits & 0xFF);
            }
            size += 3, bits = n = 0;
        } else
            bits <<= 6;
    }
    if (c == '=' && n > 1) 
        while (--n) {
            if (ptr1)
                *ptr1++ = (uint8_t) (bits >> 16);
            ++size, bits <<= 8;
        }
error:
    return size;
}

}

#endif // O3_TOOLS_MEM_H
