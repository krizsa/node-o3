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
#ifndef O3_C_TEST1_H
#define O3_C_TEST1_H

#include "o3_test_Buf.h"
#include "o3_test_tVec.h"
#include "o3_test_Str.h"
#include "o3_test_WStr.h"
#include "o3_test_Var.h"
#include "o3_test_tList.h"
#include "o3_test_tMap.h"

namespace o3 {

struct cTest1 : cScr {
    o3_begin_class(cScr)
    o3_end_class()

	o3_glue_gen()

    static o3_ext("cO3") o3_fun void testBuf()
    {
        o3::test_Buf();
    }

    static o3_ext("cO3") o3_fun void testVec()
    {
        o3::test_tVec();
    }

    static o3_ext("cO3") o3_fun void testStr()
    {
        o3::test_Str();
    }

    static o3_ext("cO3") o3_fun void testWStr()
    {
        o3::test_WStr();
    }

    static o3_ext("cO3") o3_fun void testVar()
    {
        o3::test_Var();
    }

    static o3_ext("cO3") o3_fun void testList()
    {
        o3::test_tList();
    }

    static o3_ext("cO3") o3_fun void testMap()
    {
        o3::test_tMap();
    }
};

}

#endif // O3_C_TEST1_H
