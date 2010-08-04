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
#ifndef O3_TEST_VAR_H
#define O3_TEST_VAR_H

namespace o3 {

void test_Var()
{
    o3_log("Testing Var::Var(iAlloc*)\n");
    {
        Var val;

        o3_assert(val.type() == Var::TYPE_VOID);
    }

    o3_log("Testing Var::Var(bool, iAlloc*)\n");
    {
        Var val = (bool) 0;

        o3_assert(val.type() == Var::TYPE_BOOL);
    }

    o3_log("Testing Var::Var(int32_t, iAlloc*)\n");
    {
        Var val = (int32_t) 0;

        o3_assert(val.type() == Var::TYPE_INT32);
    }

    o3_log("Testing Var::Var(int64_t, iAlloc*)\n");
    {
        Var val = (int64_t) 0;

        o3_assert(val.type() == Var::TYPE_INT64);
    }

    o3_log("Testing Var::Var(size_t, iAlloc*)\n");
    {
        Var val = (size_t) 0;

        o3_assert(val.type() == Var::TYPE_INT32);
    }

    o3_log("Testing Var::Var(double, iAlloc*)\n");
    {
        Var val = (double) 0;

        o3_assert(val.type() == Var::TYPE_DOUBLE);
    }

    o3_log("Testing Var::Var(iScr*, iAlloc*)\n");
    {
        {
            Var val = (iScr*) 0;

            o3_assert(val.type() == Var::TYPE_NULL);
        } {
            Var val = o3_new(cScrBuf)(Buf());;

            o3_assert(val.type() == Var::TYPE_SCR);
        }
    }

    o3_log("Testing Var::Var(const Str&)\n");
    {
        Var val = Str();

        o3_assert(val.type() == Var::TYPE_STR);
    }

    o3_log("Testing Var::Var(const WStr&)\n");
    {
        Var val = WStr();

        o3_assert(val.type() == Var::TYPE_WSTR);
    }

    o3_log("Testing bool Var::toBool() const\n");
    {
        {
            Var val = true;
            Var val1 = false;

            o3_assert(val.toBool());
            o3_assert(!val1.toBool());
        } {
            Var val = (int32_t) 42;
            Var val1 = (int32_t) 0;

            o3_assert(val.toBool());
            o3_assert(!val1.toBool());
        } {
            Var val = (int64_t) 42;
            Var val1 = (int64_t) 0;

            o3_assert(val.toBool());
            o3_assert(!val1.toBool());
        } {
            Var val = (double) 21.5;
            Var val1 = (double) 0;

            o3_assert(val.toBool());
            o3_assert(!val1.toBool());
        } {
            Var val = o3_new(cScrBuf)(Buf());
            Var val1 = (iScr*) 0;

            o3_assert(val.toBool());
            o3_assert(!val1.toBool());
        } {
            Var val = "true";
            Var val1 = "false";
            Var val2 = "blah";

            o3_assert(val.toBool());
            o3_assert(!val1.toBool());
            o3_assert(!val2.toBool());
        } {
            Var val = L"true";
            Var val1 = L"false";
            Var val2 = L"blah";

            o3_assert(val.toBool());
            o3_assert(!val1.toBool());
            o3_assert(!val2.toBool());
        }
    }

    o3_log("Testing int32_t Var::toInt32() const\n");
    {
        {
            Var val = true;
            Var val1 = false;

            o3_assert(val.toInt32() == 1);
            o3_assert(val1.toInt32() == 0);
        } {
            Var val = (int32_t) 42;

            o3_assert(val.toInt32() == 42);
        } {
            Var val = (int64_t) 42;

            o3_assert(val.toInt32() == 42);
        } {
            Var val = (double) 21.5;

            o3_assert(val.toInt32() == 21);
        } {
            Var val = o3_new(cScrBuf)(Buf());

            o3_assert(val.toInt32() == 0);
        } {
            Var val = "21051984";

            o3_assert(val.toInt32() == 21051984);
        } {
            Var val = L"21051984";

            o3_assert(val.toInt32() == 21051984);
        }
    }

    o3_log("Testing int64_t Var::toInt64() const\n");
    {
        {
            Var val = true;
            Var val1 = false;

            o3_assert(val.toInt64() == 1);
            o3_assert(val1.toInt64() == 0);
        } {
            Var val = (int32_t) 42;

            o3_assert(val.toInt64() == 42);
        } {
            Var val = (int64_t) 42;

            o3_assert(val.toInt64() == 42);
        } {
            Var val = (double) 21.5;

            o3_assert(val.toInt64() == 21);
        } {
            Var val = o3_new(cScrBuf)(Buf());

            o3_assert(val.toInt64() == 0);
        } {
            Var val = "21051984";

            o3_assert(val.toInt64() == 21051984);
        } {
            Var val = L"21051984";

            o3_assert(val.toInt64() == 21051984);
        }
    }

    o3_log("Testing double Var::toDouble() const\n");
    {
        {
            Var val = true;
            Var val1 = false;

            o3_assert(val.toDouble() == 1);
            o3_assert(val1.toDouble() == 0);
        } {
            Var val = (int32_t) 42;

            o3_assert(val.toDouble() == 42);
        } {
            Var val = (int64_t) 42;

            o3_assert(val.toDouble() == 42);
        } {
            Var val = 21.5;

            o3_assert(val.toDouble() == 21.5);
        } {
            Var val = o3_new(cScrBuf)(Buf());

            o3_assert(val.toDouble() == 0);
        } {
            Var val = "123.456";

            o3_assert(val.toDouble() == 123.456);
        } {
            Var val = L"123.456";

            o3_assert(val.toDouble() == 123.456);
        }
    }

    o3_log("Testing siScr Var::toScr() const\n");
    {
        {
            Var val = true;
            Var val1 = false;

            o3_assert(!val.toScr());
            o3_assert(!val1.toScr());
        } {
            Var val = (int32_t) 42;

            o3_assert(!val.toScr());
        } {
            Var val = (int64_t) 42;

            o3_assert(!val.toScr());
        } {
            Var val = 21.5;

            o3_assert(!val.toScr());
        } {
            siScr scr = o3_new(cScrBuf)(Buf());
            Var val = scr;

            o3_assert(val.toScr() == scr);
        } {
            Var val = "The quick brown fox";

            o3_assert(!val.toScr());
        } {
            Var val = L"The quick brown fox";

            o3_assert(!val.toScr());
        }
    }

    o3_log("Testing Str Var::toStr() const\n");
    {
        {
            Var val = true;
            Var val1 = false;

            o3_assert(val.toStr() == "true");
            o3_assert(val1.toStr() == "false");
        } {
            Var val = (int32_t) 21051984;

            o3_assert(val.toStr() == "21051984");
        } {
            Var val = (int64_t) 21051984;

            o3_assert(val.toStr() == "21051984");
        } {
            Var val = (double) 123.456;

            o3_assert(val.toStr() == "123.456000");
        } {
            Var val = o3_new(cScrBuf)(Buf());

            o3_assert(val.toStr() == "object");
        } {
            Var val = "The quick brown fox";

            o3_assert(val.toStr() == "The quick brown fox");
        } {
            Var val = L"The quick brown fox";

            o3_assert(val.toStr() == "The quick brown fox");
        }
    }

    o3_log("Testing WStr Var::toWStr() const\n");
    {
        {
            Var val = true;
            Var val1 = false;

            o3_assert(val.toWStr() == L"true");
            o3_assert(val1.toWStr() == L"false");
        } {
            Var val = (int32_t) 21051984;

            o3_assert(val.toWStr() == L"21051984");
        } {
            Var val = (int64_t) 21051984;

            o3_assert(val.toWStr() == L"21051984");
        } {
            Var val = (double) 123.456;

            o3_assert(val.toWStr() == L"123.456000");
        } {
            Var val = o3_new(cScrBuf)(Buf());

            o3_assert(val.toWStr() == L"object");
        } {
            Var val = L"The quick brown fox";

            o3_assert(val.toWStr() == L"The quick brown fox");
        } {
            Var val = L"The quick brown fox";

            o3_assert(val.toWStr() == L"The quick brown fox");
        }
    }
}

}

#endif // O3_TEST_VAR_H
