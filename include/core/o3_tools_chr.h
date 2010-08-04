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
#ifndef O3_TOOLS_CHR_H
#define O3_TOOLS_CHR_H

#include <ctype.h>
#include <wctype.h>

namespace o3 {

template<typename C>
inline bool chrIsBetween(C a, C b, C c)
{
    o3_trace0 trace;

    return a <= b && b <= c;
}

inline bool chrIsSpace(char c)
{
    o3_trace0 trace;

    return ::isspace(c) ? true : false;
}

inline bool chrIsSpace(wchar_t c)
{
    o3_trace0 trace;

    return ::iswspace(c) ? true : false;
}

inline bool chrIsDigit(char c)
{
	o3_trace0 trace;
	
    return ::isdigit(c) ? true : false;
}

inline bool chrIsDigit(wchar_t c)
{
	o3_trace0 trace;
	
    return ::iswdigit(c) ? true : false;
}

inline bool chrIsUpper(char c)
{
    o3_trace0 trace;

    return ::isupper(c) ? true : false;
}

inline bool chrIsUpper(wchar_t c)
{
    o3_trace0 trace;

    return ::iswupper(c) ? true : false;
}

inline bool chrIsLower(char c)
{
    o3_trace0 trace;

    return ::islower(c) ? true : false;
}

inline bool chrIsLower(wchar_t c)
{
    o3_trace0 trace;

    return ::iswlower(c) ? true : false;
}

inline char chrToUpper(char c)
{
    o3_trace0 trace;

    return (char) ::toupper(c);
}

inline wchar_t chrToUpper(wchar_t c)
{
    o3_trace0 trace;

    return ::towupper(c);
}

inline char chrToLower(char c)
{
    o3_trace0 trace;

    return (char) ::tolower(c);
}

inline wchar_t chrToLower(wchar_t c)
{
    o3_trace0 trace;

    return ::towlower(c);
}

inline char chrFromHex(int x)
{
    o3_trace0 trace;
    const char FROM_HEX[] = "0123456789ABCDEF";

    return FROM_HEX[x];
}

inline int chrToHex(char c)
{
    static char TO_HEX[256];
    o3_trace0 trace;

    if (TO_HEX[0] == 0) {
        for (size_t i = 0; i < 256; ++i)
            TO_HEX[i] = -1;
        for (size_t i = 0; i < 16; ++i) {
            char c = chrFromHex((int) i);

            TO_HEX[(int) c] = (char) i;
            TO_HEX[(int) chrToLower(c)] = (char) i;
        }
    }
    return TO_HEX[(int) c];
}

inline char chrFromBase64(int x)
{
    o3_trace0 trace;
    const char FROM_BASE64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef"
                               "ghijklmnopqrstuvwxyz0123456789+/";

    return FROM_BASE64[x];
}

template<typename C>
inline int chrToBase64(C c)
{
    static char TO_BASE64[256];
    o3_trace0 trace;

    if (TO_BASE64[0] == 0) {
        for (size_t i = 0; i < 256; ++i)
            TO_BASE64[i] = -1;
        for (size_t i = 0; i < 64; ++i) 
            TO_BASE64[(int) chrFromBase64((int) i)] = (char) i;
    }
    return TO_BASE64[(int) c];
}

}

#endif // O3_TOOLS_CHR_H
