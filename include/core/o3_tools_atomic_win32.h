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
#ifndef O3_TOOLS_ATOMIC_WIN32_H
#define O3_TOOLS_ATOMIC_WIN32_H

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace o3 {

inline int atomicTas(volatile int& x)
{
    return (int) InterlockedExchange((LONG volatile*) &x, 1);
}

inline int atomicInc(volatile int& x)
{
    return (int) InterlockedIncrement((LONG volatile*) &x); 
}

inline int atomicDec(volatile int& x)
{
    return (int) InterlockedDecrement((LONG volatile*) &x);
}

}

#endif // O3_TOOLS_ATOMIC_WIN32_H
