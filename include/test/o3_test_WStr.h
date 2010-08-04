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
#ifndef O3_TEST_W_STR_H
#define O3_TEST_W_STR_H

#include <limits.h>
#include <float.h>
#include <math.h>

namespace o3 {

inline void test_WStr()
{
    const char STR[] = "The quick brown fox jumps over the lazy dog";
    const wchar_t WSTR[] = L"The quick brown fox jumps over the lazy dog";

    o3_log("Testing static WStr WStr::fromBool(bool, iAlloc*)\n");
    {
        o3_assert(strEquals(WStr::fromBool(true).ptr(), L"true"));
        o3_assert(strEquals(WStr::fromBool(false).ptr(), L"false"));
    }

    o3_log("Testing static WStr WStr::fromInt32(int32_t, iAlloc*)\n");
    {
        const wchar_t STR[] = L"21051984";
        const wchar_t STR1[] = L"-2147483648";
        const wchar_t STR2[] = L"2147483647";

        o3_assert(strEquals(WStr::fromInt32(21051984).ptr(), STR));
        o3_assert(strEquals(WStr::fromInt32(LONG_MIN).ptr(), STR1));
        o3_assert(strEquals(WStr::fromInt32(LONG_MAX).ptr(), STR2));
    }

    o3_log("Testing static WStr WStr::fromInt64(int64_t, iAlloc*)\n");
    {
        const wchar_t STR[] = L"21051984";
        const wchar_t STR1[] = L"-9223372036854775808";
        const wchar_t STR2[] = L"9223372036854775807";

        o3_assert(strEquals(WStr::fromInt64(21051984).ptr(), STR));
        o3_assert(strEquals(WStr::fromInt64(LLONG_MIN).ptr(), STR1));
        o3_assert(strEquals(WStr::fromInt64(LLONG_MAX).ptr(), STR2));
    }

    o3_log("Testing static WStr WStr::fromDouble(double, iAlloc*)\n");
    {
        const wchar_t STR[]  = L"123.456000";
        const wchar_t STR1[] = L"0.000000";
/*        const wchar_t STR2[] = L"1797693134862315708145274237317043567980"
                               L"7056752584499659891747680315726078002853"
                               L"8760589558632766878171540458953514382464"
                               L"2343213268894641827684675467035375169860"
                               L"4991057655128207624549009038932894407586"
                               L"8508455133942304583236903222948165808559"
                               L"3321233482747978262041447231687381771809"
                               L"19299881250404026184124858368.000000";
        const wchar_t STR3[] = L"-inf";
        const wchar_t STR4[] = L"inf";*/

        o3_assert(strEquals(WStr::fromDouble(123.456).ptr(), STR));
        o3_assert(strEquals(WStr::fromDouble(DBL_MIN).ptr(), STR1));

#ifndef O3_WIN32
		o3_assert(strEquals(WStr::fromDouble(DBL_MAX).ptr(), STR2));
        o3_assert(strEquals(WStr::fromDouble(-INFINITY).ptr(), STR3));
        o3_assert(strEquals(WStr::fromDouble(INFINITY).ptr(), STR4));
#endif
    }

    o3_log("Testing static WStr WStr::fromHex(const void* ptr, size_t size)\n");
    {
/*        const wchar_t WSTR1[] = L"54 00 00 00 68 00 00 00 65 00 00 00 20 00 00 00 "
                                L"71 00 00 00 75 00 00 00 69 00 00 00 63 00 00 00 "
                                L"6B 00 00 00 20 00 00 00 62 00 00 00 72 00 00 00 "
                                L"6F 00 00 00 77 00 00 00 6E 00 00 00 20 00 00 00 "
                                L"66 00 00 00 6F 00 00 00 78 00 00 00 20 00 00 00 "
                                L"6A 00 00 00 75 00 00 00 6D 00 00 00 70 00 00 00 "
                                L"73 00 00 00 20 00 00 00 6F 00 00 00 76 00 00 00 "
                                L"65 00 00 00 72 00 00 00 20 00 00 00 74 00 00 00 "
                                L"68 00 00 00 65 00 00 00 20 00 00 00 6C 00 00 00 "
                                L"61 00 00 00 7A 00 00 00 79 00 00 00 20 00 00 00 "
                                L"64 00 00 00 6F 00 00 00 67 00 00 00 00 00 00 00";*/

#ifndef O3_WIN32
		o3_assert(strEquals(WStr::fromHex(WSTR, sizeof(WSTR)).ptr(), WSTR1));
#endif
    }

    o3_log("Testing static WStr WStr::fromBase64(const void* ptr, size_t size)\n");
    {
        const wchar_t WSTR[] = L"VGhlIHF1aWNrIGJyb3duIGZveCBqdW"
                               L"1wcyBvdmVyIHRoZSBsYXp5IGRvZwA=";
        
        o3_assert(strEquals(WStr::fromBase64(STR, sizeof(STR)).ptr(), WSTR));
    }

    o3_log("Testing explicit WStr::WStr(size_t, iAlloc*)\n");
    {
        {
            WStr str;

            o3_assert(str.capacity() == O3_AUTO_CAPACITY / sizeof(wchar_t) - 1);
            o3_assert(str.size() == 0);
        } {
            WStr str(100);

            o3_assert(str.capacity() == 127);
            o3_assert(str.size() == 0);
        }
    }

    o3_log("Testing WStr::WStr(const wchar_t*, iAlloc*)\n");
    {
        WStr str = WSTR;

        o3_assert(strEquals(str.ptr(), WSTR));
    }

    o3_log("Testing WStr::WStr(const char*, iAlloc*)\n");
    {
        WStr str = STR;

        o3_assert(strEquals(str.ptr(), WSTR));
    }

    o3_log("Testing WStr::WStr(const Buf&)\n");
    {
        {
            Buf buf;

            buf.append(WSTR, sizeof(WSTR));
            {
                WStr str = buf;

                o3_assert(!buf.unique());
                o3_assert(strEquals(str.ptr(), WSTR));
            }
        } {
            Buf buf;

            buf.append(WSTR, 32 * sizeof(wchar_t));
            {
                WStr str = buf;

                o3_assert(buf.unique());
                o3_assert(strEquals(str.ptr(), WSTR, 32));
            }
        }
    }

    o3_log("Testing bool WStr::operator==(const wchar_t*) const\n");
    {
        WStr str = L"The lazy dog jumps over";

        o3_assert(str == L"The lazy dog jumps over");
        o3_assert(!(str == L"The lazy dog jumps over the quick brown fox"));
        o3_assert(!(str == L"The quick brown fox"));
        o3_assert(!(str == WSTR));
    }

    o3_log("Testing size_t WStr::find(size_t, const wchar_t*) const\n");
    {
        WStr str = WSTR;

        o3_assert(str.find(10, L"jumps over") == 20);
        o3_assert(str.find(30, L"jumps over") == NOT_FOUND);
        o3_assert(str.find(20, L"jumps under") == NOT_FOUND);
    }

    o3_log("Testing void WStr::reserve(size_t)\n");
    {
        WStr str;

        str.reserve(64);
        o3_assert(str.capacity() == 127);
        o3_assert(str.size() == 0);
    }

    o3_log("Testing void WStr::resize(size_t)\n");
    {
        WStr str = WSTR;

        str.resize(20);
        o3_assert(str.capacity() == 63);
        o3_assert(str.size() == 20);
        o3_assert(strEquals(str.ptr(), L"The quick brown fox "));
    }

    o3_log("Testing bool WStr::toBool()\n");
    {
        o3_assert(WStr(L"true").toBool());
        o3_assert(!WStr(L"false").toBool());
        o3_assert(!WStr(L"blah").toBool());
    }

    o3_log("Testing int32_t WStr::toInt32()\n");
    {
        const wchar_t STR[] = L"21051984";
        const wchar_t STR1[] = L"-2147483648";
        const wchar_t STR2[] = L"2147483647";

        o3_assert(WStr(STR).toInt32() == 21051984);
        o3_assert(WStr(STR1).toInt32() == LONG_MIN);
        o3_assert(WStr(STR2).toInt32() == LONG_MAX);
    }

    o3_log("Testing int64_t WStr::toInt64()\n");
    {
        const wchar_t STR[] = L"21051984";
        const wchar_t STR1[] = L"-9223372036854775808";
        const wchar_t STR2[] = L"9223372036854775807";

        o3_assert(WStr(STR).toInt64() == 21051984);
        o3_assert(WStr(STR1).toInt64() == LLONG_MIN);
        o3_assert(WStr(STR2).toInt64() == LLONG_MAX);
    }

    o3_log("Testing double WStr::toDouble()\n");
    {
        const wchar_t STR[]  = L"123.456000";
        const wchar_t STR2[] = L"1797693134862315708145274237317043567980"
                               L"7056752584499659891747680315726078002853"
                               L"8760589558632766878171540458953514382464"
                               L"2343213268894641827684675467035375169860"
                               L"4991057655128207624549009038932894407586"
                               L"8508455133942304583236903222948165808559"
                               L"3321233482747978262041447231687381771809"
                               L"19299881250404026184124858368.000000";
/*        const wchar_t STR3[] = L"-inf";
        const wchar_t STR4[] = L"inf";*/

        o3_assert(WStr(STR).toDouble() == 123.456);
        // TODO: DBL_MIN
        o3_assert(WStr(STR2).toDouble() == DBL_MAX);
//        o3_assert(WStr(STR3).toDouble() == -INFINITY);
//        o3_assert(WStr(STR4).toDouble() == INFINITY);
    }

    o3_log("Testing void WStr::insert(size_t, wchar_t, size_t)\n");
    {
        const wchar_t WSTR1[] = L"The quick brown fox the lazy dog";
        const wchar_t WSTR2[] = L"The quick brown fox XXXXXXXXXXthe lazy dog";
        WStr str = WSTR1;

        str.insert(20, L'X', 10);
        o3_assert(strEquals(str.ptr(), WSTR2));
    }

    o3_log("Testing void WStr::insert(size_t, const wchar_t*)\n");
    {
        const wchar_t WSTR1[] = L"The quick brown fox the lazy dog";
        const wchar_t WSTR2[] = L"jumps over ";
        WStr str = WSTR1;

        str.insert(20, WSTR2);
        o3_assert(strEquals(str.ptr(), WSTR));
    }

    o3_log("Testing void WStr::insertf(size_t, const wchar_t*, ...)\n");
    {
        const wchar_t WSTR[] = L"The quick brown fox lazy dogs";
        const wchar_t WSTR1[] = L"The quick brown fox jumps 15 times over 3.141 lazy dogs";
        WStr str = WSTR;

        str.insertf(20, L"%ls%3d times over %.3f ", L"jumps", 15, 3.141);
        o3_assert(strEquals(str.ptr(), WSTR1));
    }

    o3_log("Testing void WStr::append(wchar_t, size_t)\n");
    {
        const wchar_t WSTR1[] = L"The quick brown fox ";
        const wchar_t WSTR2[] = L"The quick brown fox XXXXXXXXXX";
        WStr str = WSTR1;

        str.append(L'X', 10);
        o3_assert(strEquals(str.ptr(), WSTR2));
    }

    o3_log("Testing void WStr::append(const wchar_t*)\n");
    {
        const wchar_t WSTR1[] = L"The quick brown fox ";
        const wchar_t WSTR2[] = L"jumps over the lazy dog";
        WStr str = WSTR1;

        str.append(WSTR2);
        o3_assert(strEquals(str.ptr(), WSTR));
    }

    o3_log("Testing void WStr::appendf(const wchar_t*, ...)\n");
    {
        const wchar_t WSTR[] = L"The quick brown fox ";
        const wchar_t WSTR1[] = L"The quick brown fox jumps 15 times over 3.141 lazy dogs";
        WStr str = WSTR;

        str.insertf(20, L"%ls%3d times over %.3f lazy dogs", L"jumps", 15, 3.141);
        o3_assert(strEquals(str.ptr(), WSTR1));
    }

    o3_log("Testing WStr WStr::concat(const WStr&)\n");
    {
        WStr str = WStr("The quick brown fox ") + "jumps over " + "the lazy dog";

        o3_assert(strEquals(str.ptr(), WSTR));
    }

    o3_log("Testing WStr WStr::substr(size_t, size_t)\n");
    {
        o3_assert(strEquals(WStr(WSTR).substr(20, 10).ptr(), L"jumps over"));
    }

    o3_log("Testing void Str::remove(size_t, size_t)\n");
    {
        const wchar_t WSTR1[] = L"The quick brown fox the lazy dog";
        WStr str = WSTR;

        str.remove(20, 11);
        o3_assert(strEquals(str.ptr(), WSTR1));
    }

    o3_log("Testing void WStr::replace(size_t, size_t, wchar_t, size_t)\n");
    {
        // TODO
    }

    o3_log("Testing void WStr::replace(size_t, size_t, const wchar_t*)\n");
    {
        // TODO
    }

    o3_log("Testing void WStr::replacef(size_t, size_t, const wchar_t*, ...)\n");
    {
        // TODO
    }
}

}

#endif // O3_TEST_W_STR_H
