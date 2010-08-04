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
#ifndef O3_I_HTTP_H
#define O3_I_HTTP_H

namespace o3 {

o3_iid(iHttp,	0x1383b41a, 
				0xdd60, 
				0x45f5, 
				0xb2, 0x21, 0xf0, 0x38, 0x3a, 0xe0, 0x80, 0x3b);


struct iHttp : iUnk {
	enum ReadyState {
		READY_STATE_UNINITIALIZED,
		READY_STATE_LOADING,
		READY_STATE_LOADED,
		READY_STATE_INTERACTIVE,
		READY_STATE_COMPLETED
	};

	enum Method {
		METHOD_GET,
		METHOD_POST,
		METHOD_PUT
	};

	virtual ReadyState readyState() = 0;
	virtual void open(const char* method, const char* url,
		bool async = true) = 0;
	virtual void setRequestHeader(const char* name, const char* value) = 0;
	virtual void send(iCtx* ctx, const Buf& buf, bool blocking) = 0;
	virtual void send(iCtx* ctx, const Str& str, bool blocking) = 0;
	virtual Str statusText() = 0;
	virtual Str getAllResponseHeaders() = 0;
	virtual Str getResponseHeader(const char* name) = 0;
	virtual size_t bytesTotal() = 0;
	virtual size_t bytesReceived() = 0;
	virtual Buf responseBlob() = 0;
	virtual Str responseText() = 0;
	virtual void abort() = 0;
	virtual void setOnreadystatechange(Delegate) = 0;
	virtual void setOnprogress(Delegate) = 0;
};

}

#endif // O3_I_HTTP_H
