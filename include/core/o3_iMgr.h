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
#ifndef O3_I_MGR_H
#define O3_I_MGR_H

namespace o3 {

struct Trait;

template<typename C>
class tStr;

template<typename K, typename V>
class tMap;


typedef tStr<char> Str;
typedef tStr<wchar_t> WStr;

struct iCtx;
typedef siUnk (*factory_t)(iCtx*);
class Delegate;
class Buf;

o3_iid(iMgr, 0x184F9D2C,
             0xCC8D,
             0x41B9,
             0x89, 0x49, 0xAB, 0x4B, 0x43, 0xFB, 0x39, 0x90);

struct iMgr : iUnk {
    virtual Trait* extTraits(const char* name) = 0;

    virtual bool loadModule(const char *name) = 0;

    virtual void addExtTraits(Trait* traits) = 0;

    virtual siThreadPool pool() = 0;

    virtual Str root() = 0;
	
	virtual void addFactory(const Str& name, factory_t factory) = 0;

	virtual factory_t factory(const Str& name) = 0;

	virtual Str currentUrl() = 0;

	virtual void setCurrentUrl(const Str& url) = 0;

	virtual tMap<Str, int> readSettings() = 0;

	virtual void writeSettings(const tMap<Str, int>& settings) = 0;

	virtual Str allSettings() = 0;

	virtual bool writeAllSettings(const Str& settings) = 0;

	virtual Buf downloadComponent(iCtx* ctx, const Str& name, 
		Delegate onreadystatechange, Delegate onprogress) = 0;

	virtual Buf downloadUpdateInfo(iCtx* ctx) = 0;

	virtual Str latestVersion(iCtx* ctx) = 0;
};

}

#endif // O3_I_MGR_H
