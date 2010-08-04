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
#ifndef O3_I_STREAM_H
#define O3_I_STREAM_H

namespace o3 {

o3_iid(iStream, 0xFA4C77A5,
                0xC8B1,
                0x4718,
                0xB6, 0x41, 0x7F, 0x9B, 0x35, 0xD2, 0xED, 0xA8);

struct iStream : iUnk {
    virtual bool eof() = 0;

    virtual bool error() = 0;

    virtual size_t pos() = 0;

    virtual size_t setPos(size_t pos) = 0;

    virtual size_t read(void* data, size_t size) = 0;

    virtual size_t write(const void* data, size_t size) = 0;

    virtual bool flush() = 0;

    virtual bool close() = 0;

    virtual void* unwrap() = 0;

	virtual size_t size() = 0;
};

}

#endif // O3_I_STREAM_H
