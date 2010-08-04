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
#ifndef O3_I_CTX_H
#define O3_I_CTX_H

namespace o3 {

class Var;

o3_iid(iEx, 0x622E3C0B,
            0x5F4D,
            0x4229,
            0x8E, 0x89, 0xD0, 0xED, 0xCF, 0x2A, 0xC6, 0xA2);

struct iEx : iUnk {
    virtual Str message() = 0;
};

o3_iid(iCtx, 0xF12F20BB,
             0x295C,
             0x4F8B,
             0x91, 0x47, 0xC8, 0x03, 0x0D, 0x6F, 0x45, 0x37);

struct iCtx : iAlloc {
    virtual siMgr mgr() = 0;

    virtual siMessageLoop loop() = 0;

    virtual Var value(const char* key) = 0;

    virtual Var setValue(const char* key, const Var& val) = 0;

    virtual Var eval(const char* name, siEx* ex = 0) = 0;

	virtual void setAppWindow(void*) = 0;

	virtual void* appWindow() = 0;
};

}

#endif // O3_I_CTX_H
