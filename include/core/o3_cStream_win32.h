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
#ifndef O3_C_STREAM_WIN32_H
#define O3_C_STREAM_WIN32_H

#include <shared/o3_tools_win32.h>

namespace o3 {

struct cStream : cStreamBase {
    
    cStream() : m_handle(0) 
    {
    }

    cStream(void* handle) 
        : m_handle(handle) 
    {
    }

    virtual ~cStream() 
    {
        if (m_handle)
            ::CloseHandle(m_handle);
        m_handle = 0;
    } 

    o3_begin_class(cStreamBase)            
        o3_add_iface(iStream)
    o3_end_class()

	o3_glue_gen()

    HANDLE m_handle;

public:

    static siStream create(void* handle) {
        return o3_new(cStream)(handle);
    }

    void setHandle(void* handle) {
        m_handle = (HANDLE) handle;
    }

    size_t write(const void* buf, size_t nbytes) {
	    DWORD length;
        if ( ! ::WriteFile( m_handle, buf, DWORD(nbytes), &length, NULL ) )
		    return 0;

	    return (DWORD) length;
    }

    size_t read(void* buf, size_t nbytes) {
	    DWORD length;
        if ( ! ::ReadFile( m_handle, buf, DWORD(nbytes), &length, NULL ) )
		    return 0;			

	    return (size_t) length;
    }

    virtual size_t write(const char* data) {
        return write(data, strLen(data) * sizeof(char)); 
    }


    o3_get size_t size() {
	    DWORD high = 0;
        DWORD low = ::GetFileSize(m_handle, &high);	    
		if(INVALID_FILE_SIZE == low) 
		    return 0;
		
	    ULARGE_INTEGER ret;
	    ret.HighPart = high;
	    ret.LowPart = low;
	    return (size_t) ret.QuadPart;
    }

    size_t pos() {				
        return ::SetFilePointer(m_handle,0,0,FILE_CURRENT);
    }

    size_t setPos(size_t pos) {
	    LARGE_INTEGER li;
	    li.QuadPart = pos;
        li.LowPart = ::SetFilePointer(m_handle,li.LowPart,&li.HighPart,FILE_BEGIN);

	    if ( INVALID_SET_FILE_POINTER == li.LowPart )
	        return this->pos();

	    return pos;
    }

    bool eof() {
	    unsigned char buf[2];
	    if (0 == read(buf,1))
		    return true;
		
        LARGE_INTEGER li;
	    li.QuadPart = -1;
        ::SetFilePointer(m_handle,li.LowPart,&li.HighPart,FILE_CURRENT);
	    return false;
    }

    bool close() {
        BOOL res(TRUE);
        if (m_handle)
            res = ::CloseHandle(m_handle);
        m_handle = 0;
	    return res ? true : false;
    }

    bool flush() { 
        return true;
    }

    static siScr factory() {
        siScr ret = o3_new(cStream);
        return ret;
    }

    bool error(){
        return false;
    }

    void* unwrap()
    {
        return m_handle; 
    }
};

}

#endif // O3_C_STREAM_WIN32_H
