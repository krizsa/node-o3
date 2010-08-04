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
#ifndef O3_TEST_STR_H
#define O3_TEST_STR_H

#include <limits.h>
#include <float.h>
#include <math.h>

namespace o3 {

inline void test_Str()
{
    const char STR[] = "The quick brown fox jumps over the lazy dog";
    const wchar_t WSTR[] = L"The quick brown fox jumps over the lazy dog";

    o3_log("Testing static Str Str::fromBool(bool, iAlloc*)\n");
    {
        o3_assert(strEquals(Str::fromBool(true).ptr(), "true"));
        o3_assert(strEquals(Str::fromBool(false).ptr(), "false"));
    }

    o3_log("Testing static Str Str::fromInt32(int32_t, iAlloc*)\n");
    {
        const char STR[] = "21051984";
        const char STR1[] = "-2147483648";
        const char STR2[] = "2147483647";

        o3_assert(strEquals(Str::fromInt32(21051984).ptr(), STR));
        o3_assert(strEquals(Str::fromInt32(LONG_MIN).ptr(), STR1));
        o3_assert(strEquals(Str::fromInt32(LONG_MAX).ptr(), STR2));
    }

    o3_log("Testing static Str Str::fromInt64(int64_t, iAlloc*)\n");
    {
        const char STR[] = "21051984";
        const char STR1[] = "-9223372036854775808";
        const char STR2[] = "9223372036854775807";

        o3_assert(strEquals(Str::fromInt64(21051984).ptr(), STR));
        o3_assert(strEquals(Str::fromInt64(LLONG_MIN).ptr(), STR1));
        o3_assert(strEquals(Str::fromInt64(LLONG_MAX).ptr(), STR2));
    }

    o3_log("Testing static Str Str::fromDouble(double, iAlloc*)\n");
    {
        const char STR[]  = "123.456000";
        const char STR1[] = "0.000000";
/*        const char STR2[] = "1797693134862315708145274237317043567980"
                            "7056752584499659891747680315726078002853"
                            "8760589558632766878171540458953514382464"
                            "2343213268894641827684675467035375169860"
                            "4991057655128207624549009038932894407586"
                            "8508455133942304583236903222948165808559"
                            "3321233482747978262041447231687381771809"
                            "19299881250404026184124858368.000000";*/
/*        const char STR3[] = "-inf";*/
/*        const char STR4[] = "inf";*/

        o3_assert(strEquals(Str::fromDouble(123.456).ptr(), STR));
        o3_assert(strEquals(Str::fromDouble(DBL_MIN).ptr(), STR1));

#ifndef O3_WIN32        
		o3_assert(strEquals(Str::fromDouble(DBL_MAX).ptr(), STR2));
        o3_assert(strEquals(Str::fromDouble(-INFINITY).ptr(), STR3));
        o3_assert(strEquals(Str::fromDouble(INFINITY).ptr(), STR4));
#endif
    }

    o3_log("Testing static Str Str::fromHex(const void* ptr, size_t size)\n");
    {
/*        const char STR[] = "54 00 00 00 68 00 00 00 65 00 00 00 20 00 00 00 "
                           "71 00 00 00 75 00 00 00 69 00 00 00 63 00 00 00 "
                           "6B 00 00 00 20 00 00 00 62 00 00 00 72 00 00 00 "
                           "6F 00 00 00 77 00 00 00 6E 00 00 00 20 00 00 00 "
                           "66 00 00 00 6F 00 00 00 78 00 00 00 20 00 00 00 "
                           "6A 00 00 00 75 00 00 00 6D 00 00 00 70 00 00 00 "
                           "73 00 00 00 20 00 00 00 6F 00 00 00 76 00 00 00 "
                           "65 00 00 00 72 00 00 00 20 00 00 00 74 00 00 00 "
                           "68 00 00 00 65 00 00 00 20 00 00 00 6C 00 00 00 "
                           "61 00 00 00 7A 00 00 00 79 00 00 00 20 00 00 00 "
                           "64 00 00 00 6F 00 00 00 67 00 00 00 00 00 00 00";*/
#ifndef O3_WIN32
        o3_assert(strEquals(Str::fromHex(WSTR, sizeof(WSTR)).ptr(), STR));
#endif
	}

    o3_log("Testing static Str Str::fromBase64(const void* ptr, size_t size)\n");
    {
        const char STR1[] = "VGhlIHF1aWNrIGJyb3duIGZveCBqdW"
                            "1wcyBvdmVyIHRoZSBsYXp5IGRvZwA=";
        
        o3_assert(strEquals(Str::fromBase64(STR, sizeof(STR)).ptr(), STR1));
    }

    o3_log("Testing explicit Str::Str(size_t, iAlloc*)\n");
    {
        {
            Str str;

            o3_assert(str.capacity() == O3_AUTO_CAPACITY / sizeof(char) - 1);
            o3_assert(str.size() == 0);
        } {
            Str str(100);

            o3_assert(str.capacity() == 127);
            o3_assert(str.size() == 0);
        }
    }

    o3_log("Testing Str::Str(const char*, iAlloc*)\n");
    {
        Str str = STR;

        o3_assert(strEquals(str.ptr(), STR));
    }

    o3_log("Testing Str::Str(const wchar_t*, iAlloc*)\n");
    {
        Str str = WSTR;

        o3_assert(strEquals(str.ptr(), STR));
    }

    o3_log("Testing Str::Str(const Buf&)\n");
    {
        {
            Buf buf;

            buf.append(STR, sizeof(STR));
            {
                Str str = buf;

                o3_assert(!buf.unique());
                o3_assert(strEquals(str.ptr(), STR));
            }
        } {
            Buf buf;

            buf.append(STR, 32 * sizeof(char));
            {
                Str str = buf;

                o3_assert(buf.unique());
                o3_assert(strEquals(str.ptr(), STR, 32));
            }
        }
    }

    o3_log("Testing bool Str::operator==(const char*) const\n");
    {
        Str str = "The lazy dog jumps over";

        o3_assert(str == "The lazy dog jumps over");
        o3_assert(!(str == "The lazy dog jumps over the quick brown fox"));
        o3_assert(!(str == "The quick brown fox"));
        o3_assert(!(str == STR));
    }

    o3_log("Testing size_t Str::find(size_t, const char*) const\n");
    {
        Str str = STR;

        o3_assert(str.find(10, "jumps over") == 20);
        o3_assert(str.find(30, "jumps over") == NOT_FOUND);
        o3_assert(str.find(20, "jumps under") == NOT_FOUND);
    }

    o3_log("Testing void Str::reserve(size_t)\n");
    {
        Str str;

        str.reserve(64);
        o3_assert(str.capacity() == 127);
        o3_assert(str.size() == 0);
    }

    o3_log("Testing void Str::resize(size_t)\n");
    {
        Str str = STR;

        str.resize(20);
        o3_assert(str.capacity() == 63);
        o3_assert(str.size() == 20);
        o3_assert(strEquals(str.ptr(), "The quick brown fox "));
    }

    o3_log("Testing bool Str::toBool()\n");
    {
        o3_assert(Str("true").toBool());
        o3_assert(!Str("false").toBool());
        o3_assert(!Str("blah").toBool());
    }

    o3_log("Testing int32_t Str::toInt32()\n");
    {
        const char STR[] = "21051984";
        const char STR1[] = "-2147483648";
        const char STR2[] = "2147483647";

        o3_assert(Str(STR).toInt32() == 21051984);
        o3_assert(Str(STR1).toInt32() == LONG_MIN);
        o3_assert(Str(STR2).toInt32() == LONG_MAX);
    }

    o3_log("Testing int64_t Str::toInt64()\n");
    {
        const char STR[] = "21051984";
        const char STR1[] = "-9223372036854775808";
        const char STR2[] = "9223372036854775807";

        o3_assert(Str(STR).toInt64() == 21051984);
        o3_assert(Str(STR1).toInt64() == LLONG_MIN);
        o3_assert(Str(STR2).toInt64() == LLONG_MAX);
    }

    o3_log("Testing double Str::toDouble()\n");
    {
        const char STR[]  = "123.456000";
        const char STR2[] = "1797693134862315708145274237317043567980"
                            "7056752584499659891747680315726078002853"
                            "8760589558632766878171540458953514382464"
                            "2343213268894641827684675467035375169860"
                            "4991057655128207624549009038932894407586"
                            "8508455133942304583236903222948165808559"
                            "3321233482747978262041447231687381771809"
                            "19299881250404026184124858368.000000";
/*        const char STR3[] = "-inf";
        const char STR4[] = "inf";*/

        o3_assert(Str(STR).toDouble() == 123.456);
        // TODO: DBL_MIN
        o3_assert(Str(STR2).toDouble() == DBL_MAX);
        //o3_assert(Str(STR3).toDouble() == -INFINITY);
        //o3_assert(Str(STR4).toDouble() == INFINITY);
    }

    o3_log("Testing void Str::insert(size_t, char, size_t)\n");
    {
        const char STR1[] = "The quick brown fox the lazy dog";
        const char STR2[] = "The quick brown fox XXXXXXXXXXthe lazy dog";
        Str str = STR1;

        str.insert(20, 'X', 10);
        o3_assert(strEquals(str.ptr(), STR2));
    }

    o3_log("Testing void Str::insert(size_t, const char*)\n");
    {
        const char STR1[] = "The quick brown fox the lazy dog";
        const char STR2[] = "jumps over ";
        Str str = STR1;

        str.insert(20, STR2);
        o3_assert(strEquals(str.ptr(), STR));
    }

    o3_log("Testing void Str::insertf(size_t, const char*, ...)\n");
    {
        const char STR[] = "The quick brown fox lazy dogs";
        const char STR1[] = "The quick brown fox jumps 15 times over 3.141 lazy dogs";
        Str str = STR;

        str.insertf(20, "%s%3d times over %.3f ", "jumps", 15, 3.141);
        o3_assert(strEquals(str.ptr(), STR1));
    }

    o3_log("Testing void Str::append(char, size_t)\n");
    {
        const char STR1[] = "The quick brown fox ";
        const char STR2[] = "The quick brown fox XXXXXXXXXX";
        Str str = STR1;

        str.append('X', 10);
        o3_assert(strEquals(str.ptr(), STR2));
    }

    o3_log("Testing void Str::append(const char*)\n");
    {
        const char STR1[] = "The quick brown fox ";
        const char STR2[] = "jumps over the lazy dog";
        Str str = STR1;

        str.append(STR2);
        o3_assert(strEquals(str.ptr(), STR));
    }

    o3_log("Testing void Str::appendf(const char*, ...)\n");
    {
        const char STR[] = "The quick brown fox ";
        const char STR1[] = "The quick brown fox jumps 15 times over 3.141 lazy dogs";
        Str str = STR;

        str.insertf(20, "%s%3d times over %.3f lazy dogs", "jumps", 15, 3.141);
        o3_assert(strEquals(str.ptr(), STR1));
    }

    o3_log("Testing Str Str::concat(const Str&)\n");
    {
        Str str = Str("The quick brown fox ") + "jumps over " + "the lazy dog";

        o3_assert(strEquals(str.ptr(), STR));
    }

    o3_log("Testing Str Str::substr(size_t, size_t)\n");
    {
        o3_assert(strEquals(Str(STR).substr(20, 10).ptr(), "jumps over"));
    }

    o3_log("Testing void Str::remove(size_t, size_t)\n");
    {
        const char STR1[] = "The quick brown fox the lazy dog";
        Str str = STR;

        str.remove(20, 11);
        o3_assert(strEquals(str.ptr(), STR1));
    }

    o3_log("Testing void Str::replace(size_t, size_t, char, size_t)\n");
    {
        // TODO
    }

    o3_log("Testing void Str::replace(size_t, size_t, const char*)\n");
    {
        // TODO
    }

    o3_log("Testing void Str::replacef(size_t, size_t, const char*, ...)\n");
    {
        // TODO
    }
}

}

#endif // O3_TEST_STR_H
