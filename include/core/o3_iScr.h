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
#ifndef O3_I_SCR_H
#define O3_I_SCR_H

namespace o3 {

o3_iid(iScr, 0xFE84E8F8,
             0x22ED,
             0x4886,
             0xA9, 0xF1, 0x11, 0xD9, 0xBD, 0x14, 0x81, 0x7D);

struct iScr : iUnk {
    enum Access {
        ACCESS_CALL,
        ACCESS_GET,
        ACCESS_SET,
        ACCESS_DEL
    };

    virtual int enumerate(iCtx* ctx, int index = -1) = 0;

    virtual Str name(iCtx* ctx, int index) = 0;

    virtual int resolve(iCtx* ctx, const char* name, bool set = false) = 0;

    virtual siEx invoke(iCtx* ctx, Access access, int index, int argc,
                        const Var* argv, Var* rval) = 0;
};

}

#endif // O3_I_SCR_H
