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
#ifndef O3_I_FS_H
#define O3_I_FS_H

namespace o3 {

o3_iid(iFs, 0x9F96A731,
            0x53EC,
            0x4E42,
            0xA5, 0x75, 0xFB, 0x5D, 0xD3, 0x43, 0xB3, 0xB4);

struct iFs : iUnk {

	enum Type {
		TYPE_INVALID,
		TYPE_DIR,
		TYPE_FILE,
		TYPE_LINK,
		TYPE_NOTFOUND
	};

	virtual bool valid() = 0;

	virtual bool exists() = 0;

	virtual Type type() = 0;

	virtual bool isDir() = 0;

	virtual bool isFile() = 0;

	virtual bool isLink() = 0;

	virtual int64_t accessedTime() = 0;

	virtual int64_t modifiedTime() = 0;

	virtual int64_t createdTime() = 0;

	virtual size_t size() = 0;

	virtual Str path() = 0;

	virtual Str name() = 0;

	virtual Str setName(const char* name, siEx* ex=0) = 0;

	virtual siFs get(const char* path) = 0;

	virtual tVec<siFs> children() = 0;

	virtual bool createDir() = 0;

	virtual bool createFile() = 0;

	virtual bool createLink(iFs* tos) = 0;

	virtual bool remove(bool deep = true) = 0;

	virtual siFs copy(iFs* to, siEx* ex=0) = 0;

	virtual siFs move(iFs* to, siEx* ex=0) = 0;	

	virtual siStream open(const char* mode, siEx* ex = 0) = 0;

	virtual Buf blob() = 0;

	virtual Buf setBlob(const Buf& buf) = 0;

	virtual siStream setBlob(iStream* stream, siEx* ex) = 0;

	virtual Str data() = 0;

	virtual Str setData(const Str& str) = 0;

	virtual siScr onchange() = 0;

	virtual siScr setOnchange(iCtx* ctx, iScr* scr) = 0;

	virtual void openDoc() = 0;
};

}

#endif // O3_I_FS_H
