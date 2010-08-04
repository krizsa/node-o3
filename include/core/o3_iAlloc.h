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
#ifndef O3_I_ALLOC_H
#define O3_I_ALLOC_H

namespace o3 {

o3_iid(iAlloc, 0x10C3D9DB,
               0x0719,
               0x488B,
               0xB2, 0x9E, 0x38, 0x7D, 0x37, 0x38, 0xDA, 0x1C);

struct iAlloc : iUnk {
    virtual void* alloc(size_t size) = 0;

    virtual void free(void* ptr) = 0;
};

}

#endif // O3_I_ALLOC_H
