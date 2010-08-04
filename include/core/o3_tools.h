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
#ifndef O3_TOOLS_H
#define O3_TOOLS_H

#include <stdarg.h>

namespace o3 {

template<typename T>
inline T min(T x, T y)
{
    o3_trace0 trace;

    return x < y ? x : y;
}

template<typename T>
inline T max(T x, T y)
{
    o3_trace0 trace;

    return x < y ? y : x;
}

template<typename T>
inline void swap(T& x, T& y)
{
    o3_trace0 trace;
    uint8_t z[sizeof(T)];
 
    memCopy(z, &x, sizeof(T));
    memCopy(&x, &y, sizeof(T));
    memCopy(&y, z, sizeof(T));    
}

__inline int DoubleToInt(double d)
{
	const double magic = 6755399441055744.0; // 2^51 + 2^52
	double tmp = (d-0.5) + magic;
	return *(int*) &tmp;
}

}

#include "o3_tools_atomic.h"
#include "o3_tools_chr.h"
#include "o3_tools_mem.h"
#include "o3_tools_str.h"

#endif // O3_TOOLS_H
