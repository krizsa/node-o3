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
#ifndef O3_I_BUF_H
#define O3_I_BUF_H

namespace o3 {

class Buf;

o3_iid(iBuf, 0xFEBA983C,
             0xA73D,
             0x408B,
             0x85, 0x2E, 0x2A, 0x4E, 0x30, 0x84, 0xE5, 0xAB);

struct iBuf : iUnk {
    virtual Buf& unwrap() = 0;
};

}

#endif // O3_I_BUF_H
