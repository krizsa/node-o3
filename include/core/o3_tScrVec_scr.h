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
#ifndef O3_T_SCR_VEC_SCR_H
#define O3_T_SCR_VEC_SCR_H

    Trait* select()
    {
        return clsTraits();
    }

    static Trait* clsTraits()
    {
        static Trait TRAITS[] = {
            {   0,  Trait::TYPE_BEGIN,  "tScrVec",  0,                  0,          0,  cScr::clsTraits()   },
            {   0,  Trait::TYPE_GET,    "tScrVec",  "length",           clsInvoke,  0,  0                   },
            {   0,  Trait::TYPE_SET,    "tScrVec",  "length",           clsInvoke,  1,  0                   },
            {   1,  Trait::TYPE_FUN,    "tScrVec",  "__enumerator__",   clsInvoke,  2,  0                   },
            {   2,  Trait::TYPE_FUN,    "tScrVec",  "__query__",        clsInvoke,  3,  0                   },
            {   3,  Trait::TYPE_FUN,    "tScrVec",  "__getter__",       clsInvoke,  4,  0                   },
            {   4,  Trait::TYPE_FUN,    "tScrVec",  "__setter__",       clsInvoke,  5,  0                   },
            {   5,  Trait::TYPE_FUN,    "tScrVec",  "__deleter__",      clsInvoke,  6,  0                   },
            {   0,  Trait::TYPE_END,    0,          0,                  0,          0,  0                   }
        };

        return TRAITS;
    }

    static siEx clsInvoke(iScr* pthis, iCtx* ctx, int index, int,
                          const Var* argv, Var* rval)
    {
        tScrVec* pthis1 = (tScrVec*) pthis;

        switch (index) {
        case 0:
            *rval = pthis1->length();
            break;
        case 1:
            *rval = pthis1->setLength(argv[0].toInt32());
            break;
        case 2:
            *rval = pthis1->__enumerator__(argv[0].toInt32());
            break;
        case 3:
            *rval = pthis1->__query__(argv[0].toInt32());
            break;
        case 4:
            *rval = pthis1->__getter__(ctx, argv[0].toInt32());
            break;
        case 5:
            *rval = pthis1->__setter__(argv[0].toInt32(), argv[1]);
            break;
        case 6:
            *rval = pthis1->__deleter__(argv[0].toInt32());
            break;
        }
        return 0;
    }

#endif // O3_T_SCR_VEC_SCR_H
