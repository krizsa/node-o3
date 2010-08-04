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
#ifndef O3_C_HTTP1_H
#define O3_C_HTTP1_H

#define CURL_STATICLIB
#include <curl/curl.h>

namespace o3 {

struct cHttp1 : cScr, iHttp {

    static size_t read(void* ptr, size_t size, size_t nmemb, void *stream)
    {
        cHttp1* pthis = (cHttp1*) stream;
        Lock lock(pthis->m_mutex);
        memCopy(ptr, pthis->m_ptr, size = min(size * nmemb, pthis->m_size));
        pthis->m_ptr += size;
        pthis->m_size -= size;
        return size;
    }

    static size_t header(void *ptr, size_t size, size_t nmemb, void *stream)
    {
        cHttp1* pthis = (cHttp1*) stream;
        Lock lock(pthis->m_mutex);
        Str header = (const char*) ptr;
        size_t pos = header.find(":");
        if (pos != NOT_FOUND) {
            header[pos] = 0;
            pthis->m_response_headers[header] = header.ptr() + pos + 2;
        }
        return nmemb * size;
    }

    static size_t write(void *ptr, size_t size, size_t nmemb, void *stream)
    {
        cHttp1* pthis = (cHttp1*) stream;
        Lock lock(pthis->m_mutex);

        if (pthis->m_state == READY_STATE_LOADING) {
            pthis->m_state = READY_STATE_LOADED;
            pthis->m_ctx->loop()->post(Delegate(pthis,
                                                &cHttp1::callOnReadystatechange),
                                       o3_cast pthis);
            pthis->m_event->wait(pthis->m_mutex);
        }

        pthis->m_response_body.append(ptr, size = size * nmemb);
        if (!pthis->m_pending) {
            pthis->m_pending = true;
            pthis->m_ctx->loop()->post(Delegate(pthis, &cHttp1::callOnprogress),
                                       o3_cast pthis);
        }
        if (pthis->m_state == READY_STATE_LOADED) {
            pthis->m_state = READY_STATE_INTERACTIVE;
            pthis->m_ctx->loop()->post(Delegate(pthis,
                                               &cHttp1::callOnReadystatechange),
                                       o3_cast pthis);
            pthis->m_event->wait(pthis->m_mutex);
        }
        return size;
    }

    CURL* m_handle;
    siMutex m_mutex;
    ReadyState m_state;
    Method m_method;
    Str m_url;
    bool m_async;
    tMap<Str, Str> m_request_headers;
    Buf m_request_body;
    uint8_t* m_ptr;
    size_t m_size;
    tMap<Str, Str> m_response_headers;
    Buf m_response_body;
    siCtx m_ctx;
    siScr m_onreadystatechange;
    siScr m_onprogress;
	Delegate m_dg_onreadystatechange;
	Delegate m_dg_onprogress;
	siEvent m_event;
    bool m_pending;

    cHttp1()
    {
        m_handle = curl_easy_init();
        m_mutex = g_sys->createMutex();
        m_state = READY_STATE_UNINITIALIZED;
        m_event = g_sys->createEvent();
        m_pending = false;
    }

    ~cHttp1()
    {
        curl_easy_cleanup(m_handle);
    }

    o3_begin_class(cScr)
		o3_add_iface(iHttp)
    o3_end_class()

    o3_glue_gen()

    o3_enum("ReadyState",
        READY_STATE_UNINITIALIZED,
        READY_STATE_LOADING,
        READY_STATE_LOADED,
        READY_STATE_INTERACTIVE,
        READY_STATE_COMPLETED);

    static o3_ext("cO3") o3_fun siScr http()
    {
        o3_trace3 trace;

        return o3_new(cHttp1)();
    }

	static siUnk factory(iCtx*)
	{
		return http();
	}

    virtual o3_get ReadyState readyState()
    {
        o3_trace3 trace;
        Lock lock(m_mutex);

        return m_state;
    }

    virtual o3_fun void open(const char* method, const char* url,
                             bool async = true)
    {
        o3_trace3 trace;
		
        m_state = READY_STATE_LOADING;
        if (strEquals(method, "GET"))
            m_method = METHOD_GET;
        else if (strEquals(method, "POST"))
            m_method = METHOD_POST;
        else if (strEquals(method, "PUT"))
            m_method = METHOD_PUT;
        m_url = url;
        m_async = async;
		m_response_body = Buf(1024);
    }

    virtual o3_fun void setRequestHeader(const char* name, const char* value)
    {
        o3_trace3 trace;

        if (*value)
            m_request_headers[name] = value;
        else
            m_request_headers.remove(name);
    }

	virtual void send(iCtx* ctx, const Buf& buf, bool blocking)
	{
		o3_trace3 trace;

		m_request_body = buf;
		m_ptr = (uint8_t*) buf.ptr();
		m_size = buf.size();
		m_ctx = ctx;
		if (blocking)
			perform(o3_cast this);
		else
			m_ctx->mgr()->pool()->post(Delegate(this, &cHttp1::perform),
			o3_cast this);
	}

    virtual o3_fun void send(iCtx* ctx, const Buf& buf)
    {
        o3_trace3 trace;

        send(ctx,buf,!m_async);
    }

    virtual o3_fun void send(iCtx* ctx, const Str& str)
    {
        o3_trace3 trace;

        send(ctx,str,!m_async);
    }

	virtual void send(iCtx* ctx, const Str& str, bool blocking)
	{
		o3_trace3 trace;
		Buf buf(str.size());

		buf.append(str.ptr(), str.size());
		return send(ctx, buf, blocking);
	}

    virtual o3_get Str statusText()
    {
        return Str::fromInt32(statusCode());
    }

    virtual o3_get int statusCode()
    {
        long code;

        curl_easy_getinfo(m_handle, CURLINFO_RESPONSE_CODE, &code);
        return code;
    }

    virtual o3_fun Str getAllResponseHeaders()
    {
        o3_trace3 trace;
        Lock lock(m_mutex);
        Str headers;

        for (tMap<Str, Str>::ConstIter i = m_response_headers.begin();
             i != m_response_headers.end(); ++i)
            headers += i->key + ":" + i->val;
        return headers;
    }

    virtual o3_fun Str getResponseHeader(const char* name)
    {
        o3_trace3 trace;
        Lock lock(m_mutex);

        return m_response_headers[name];
    }

    virtual o3_get size_t bytesTotal()
    {
        o3_trace3 trace;
        tMap<Str, Str>::ConstIter iter;

        iter = m_response_headers.find("Content-Length");
        if (iter != m_response_headers.end())
            return iter->val.toInt32();
        return (size_t) -1;
    }

    virtual o3_get size_t bytesReceived()
    {
        o3_trace3 trace;
        Lock lock(m_mutex);

        return m_response_body.size();
    }

    virtual o3_get Buf responseBlob()
    {
        o3_trace3 trace;
        Lock lock(m_mutex);

        return m_response_body;
    }

    virtual o3_get Str responseText()
    {
        o3_trace3 trace;
        Lock lock(m_mutex);

        return Str(m_response_body);
    }

    virtual o3_fun void abort()
    {
        o3_trace3 trace;
    }

    virtual o3_get siScr onreadystatechange()
    {
        o3_trace3 trace;
        Lock lock(m_mutex);

        return m_onreadystatechange;
    }

	virtual void setOnreadystatechange(Delegate onreadystatechange)
	{
		o3_trace3 trace;
		Lock lock(m_mutex);

		m_dg_onreadystatechange = onreadystatechange;
	}

    virtual o3_set siScr setOnreadystatechange(iCtx* ctx, iScr* onreadystatechange)
    {
        o3_trace3 trace;
		setOnreadystatechange(Delegate(ctx, onreadystatechange));

		Lock lock(m_mutex);
        return m_onreadystatechange = onreadystatechange;
    }

    virtual o3_get siScr onprogress()
    {
        o3_trace3 trace;
        Lock lock(m_mutex);

        return m_onprogress;
    }

	virtual void setOnprogress(Delegate onprogress)
	{
		o3_trace3 trace;
		Lock lock(m_mutex);

		m_dg_onprogress = onprogress;
	}

    virtual o3_set siScr setOnprogress(iCtx* ctx, iScr* onprogress)
    {
        o3_trace3 trace;
		setOnprogress(Delegate(ctx, onprogress));

        Lock lock(m_mutex);
        return m_onprogress = onprogress;
    }

    void perform(iUnk*)
    {
        curl_slist *slist = 0;

		curl_easy_reset(m_handle);

        if (m_async)
            m_mutex->lock();
        switch (m_method) {
        case METHOD_GET:
            break;
        case METHOD_POST:
            curl_easy_setopt(m_handle, CURLOPT_POST, true);
            curl_easy_setopt(m_handle, CURLOPT_READFUNCTION, read);
            curl_easy_setopt(m_handle, CURLOPT_READDATA, this);
            break;
        case METHOD_PUT:
            curl_easy_setopt(m_handle, CURLOPT_UPLOAD, true);
            curl_easy_setopt(m_handle, CURLOPT_READFUNCTION, read);
            curl_easy_setopt(m_handle, CURLOPT_READDATA, this);
            break;
        }
        //if (m_async)
        //    m_mutex->unlock();
        //curl_easy_perform(m_handle);
        //curl_slist_free_all(slist);
        curl_easy_setopt(m_handle, CURLOPT_URL, m_url.ptr());
        for (tMap<Str, Str>::ConstIter i = m_request_headers.begin();
             i != m_request_headers.end(); ++i)
            slist = curl_slist_append(slist, (i->key + ":" + i->val).ptr());
        curl_easy_setopt(m_handle, CURLOPT_HTTPHEADER, slist);
        curl_easy_setopt(m_handle, CURLOPT_HEADERFUNCTION, header);
        curl_easy_setopt(m_handle, CURLOPT_WRITEHEADER, this);
        curl_easy_setopt(m_handle, CURLOPT_WRITEFUNCTION, write);
        curl_easy_setopt(m_handle, CURLOPT_WRITEDATA, this);

        if (m_async)
            m_mutex->unlock();
        curl_easy_perform(m_handle);
        curl_slist_free_all(slist);
        m_state = READY_STATE_COMPLETED;
        m_ctx->loop()->post(Delegate(this, &cHttp1::callOnReadystatechange),
                            o3_cast this);

		m_mutex->lock();
        m_event->wait(m_mutex);
		m_mutex->unlock();
    }

    void callOnReadystatechange(iUnk*)
    {
        Delegate fun;

        {
			// copy the delegate so we don't have to keep the mutex
			// locked for the whole callback
            Lock lock(m_mutex);

            fun = m_dg_onreadystatechange;
        }
        fun(this);
        m_event->signal();
    }

    void callOnprogress(iUnk*)
    {
        Delegate fun;

        {
            Lock lock(m_mutex);

            fun = m_dg_onprogress;
        }
        fun(this);
        m_pending = false;
    }


	o3_fun siFs responseOpen(iFs* parent) 
	{
		// this function is implemented in the version in the private repository only
		return siFs();
	}

};

}

#endif // O3_C_HTTP1_H
