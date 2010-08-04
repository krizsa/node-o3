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

namespace o3 {    

    DEFINE_GUID(IID_IProtocolIE, 
	0xdc5e1179, 0x4ea, 0x4848, 0xb0, 0x85, 0x2c, 0x23, 0xa7, 0xe1, 0x50, 0x31);

    struct IProtocolIE : IUnknown
    {    
    };

    mscom_ptr(IInternetProtocolSink);
    mscom_ptr(IProtocolIE);
	mscom_ptr(IClassFactory);

    struct ProtocolIE
        : IInternetProtocol
//        , IInternetProtocolRoot
        , IInternetProtocolInfo
        , IProtocolIE
        , IClassFactory
    {
        ProtocolIE(iCtx* ctx)
        {
            Var v = ctx->value("protocolHandler");
            m_protocol_handler = v.toScr();
            if (!m_protocol_handler) {
                m_protocol_handler = o3_new(cProtocol1)(ctx);
                v = siScr(m_protocol_handler);
                ctx->setValue("protocolHandler",v); 
            }
        }

        ProtocolIE(iProtocol* handler)
            :   m_protocol_handler(handler)
        {        
        }

        mscom_begin(ProtocolIE)
            mscom_add_interface(IProtocolIE)
			mscom_add_interface(IInternetProtocol)
			mscom_add_interface(IInternetProtocolRoot)
            mscom_add_interface(IInternetProtocolInfo)
            mscom_add_interface(IClassFactory)
		mscom_end();

        siStream                    m_stream;
        SIInternetProtocolSink      m_proto_sink;
        siProtocol                  m_protocol_handler;        

		static void registerProtocol(IProtocolIE* protocol_handler) 
		{
			IInternetSecurityManager *pSecurityManager = NULL;
			if (SUCCEEDED(CoCreateInstance( CLSID_InternetSecurityManager, NULL, 
				CLSCTX_INPROC_SERVER,IID_IInternetSecurityManager,(void **)&pSecurityManager )))
			{
				/*HRESULT res =*/ pSecurityManager->SetZoneMapping(URLZONE_TRUSTED,L"o3",SZM_CREATE);
				pSecurityManager->Release();
			}

			IInternetSession *ses;
			CoInternetGetSession( 0, &ses, 0 );			
			
			SIClassFactory factory = SIProtocolIE(protocol_handler);
			ses->RegisterNameSpace( factory.ptr(), 
				IID_IInternetProtocol /*IID_IProtocolIE*/, L"o3", 0 ,0 ,0 );
			ses->Release();	  
		}

		static void unregisterProtocol(IProtocolIE* protocol_handler) 
		{
			IInternetSession *ses;
			CoInternetGetSession( 0, &ses, 0 );			

			SIClassFactory factory = SIProtocolIE(protocol_handler);
			ses->UnregisterNameSpace(factory.ptr(), L"o3");
			ses->Release();
		}
		

        HRESULT ClassImplementsCategory( const CLSID& /*clsid*/, const CATID& /*catid*/ )
        {
            // TODO: double check this...
            return S_OK;
        }

        HRESULT STDMETHODCALLTYPE CreateInstance( IUnknown __RPC_FAR* /*pUnkOuter*/, 
            REFIID riid, void __RPC_FAR *__RPC_FAR *ppvObject)
	    {
            ProtocolIE* instance = o3_new(ProtocolIE)(m_protocol_handler);

		    if( instance->QueryInterface( riid, ppvObject ) != S_OK )
		    {
			    delete instance;
			    return E_NOINTERFACE;
		    }
		    return S_OK;
	    }

	    HRESULT STDMETHODCALLTYPE LockServer(BOOL /*fLock*/) 
        {
            return S_OK;
        }

	    HRESULT STDMETHODCALLTYPE Start( LPCWSTR szUrl,
            IInternetProtocolSink __RPC_FAR *pOIProtSink,
            IInternetBindInfo __RPC_FAR *pOIBindInfo,
            DWORD grfPI, DWORD /*dwReserved*/ )
        {
		    if (grfPI & PI_PARSE_URL) {
			    return S_OK;
		    }    

            WStr wurl(szUrl);
            Str url(wurl),
                path = m_protocol_handler->processURL(url);

            m_stream = m_protocol_handler->start(path);
            if (!m_stream)
                return INET_E_INVALID_URL;

            // TODO: it is not yet decided to put a size property on the stream component/iface
            // or not... this should be cleaned after the final decision has been made
            size_t size = ((cStream*)(m_stream.ptr()))->size();
            m_proto_sink = pOIProtSink;

		    DWORD dwBindf;
		    BINDINFO udtBindInfo;
		    udtBindInfo.cbSize = sizeof(BINDINFO);
		    pOIBindInfo->GetBindInfo(&dwBindf, &udtBindInfo);
		    ::ReleaseBindInfo(&udtBindInfo);

            if(!(dwBindf&BINDF_ASYNCHRONOUS)) {
                WStr wpath(path);
    			pOIProtSink->ReportProgress(BINDSTATUS_FINDINGRESOURCE, wpath);
	    		pOIProtSink->ReportProgress(BINDSTATUS_CONNECTING, wpath);
		    	pOIProtSink->ReportProgress(BINDSTATUS_SENDINGREQUEST, wpath);
			    pOIProtSink->ReportProgress(BINDSTATUS_CACHEFILENAMEAVAILABLE, wpath);            
            }else{
			    pOIProtSink->ReportProgress(BINDSTATUS_FINDINGRESOURCE, szUrl);
			    pOIProtSink->ReportProgress(BINDSTATUS_CONNECTING, szUrl);
			    pOIProtSink->ReportProgress(BINDSTATUS_SENDINGREQUEST, szUrl);            
            }

            pOIProtSink->ReportData(BSCF_FIRSTDATANOTIFICATION
                | BSCF_LASTDATANOTIFICATION | BSCF_DATAFULLYAVAILABLE, size,0);

            return S_OK;
        }

	    HRESULT STDMETHODCALLTYPE Continue( PROTOCOLDATA __RPC_FAR* /*pProtocolData*/ )
        {
            return S_OK;
        }

	    HRESULT STDMETHODCALLTYPE Abort( HRESULT /*hrReason*/, DWORD /*dwOptions*/ )
	    {
		    m_proto_sink->ReportResult(S_OK,S_OK,NULL);
		    return S_OK;
	    }

	    HRESULT STDMETHODCALLTYPE Terminate( DWORD /*dwOptions*/ )
        {
            m_stream = 0;
            m_proto_sink = 0;
            return S_OK;
        }

	    HRESULT STDMETHODCALLTYPE Suspend()
        {
            return E_NOTIMPL;
        }
    	
        HRESULT STDMETHODCALLTYPE Resume()
        {
            return E_NOTIMPL;
        }
    	
        HRESULT STDMETHODCALLTYPE Read( void __RPC_FAR *pv, ULONG cb, 
            ULONG __RPC_FAR *pcbRead)
        {
            if (!m_stream)
                return S_FALSE;

            size_t read = m_stream->read(pv, cb);
            *pcbRead = read;
            return S_OK;
        }

	    HRESULT STDMETHODCALLTYPE Seek( LARGE_INTEGER dlibMove, 
            DWORD /*dwOrigin*/, ULARGE_INTEGER __RPC_FAR *plibNewPosition)
        {
            if(!m_stream)
                return S_FALSE;
    		
	    	plibNewPosition->QuadPart = m_stream->setPos((size_t) dlibMove.QuadPart);
		    return S_OK;
        }

        HRESULT STDMETHODCALLTYPE LockRequest( DWORD /*dwOptions*/)
        {
            return S_OK;
        }

	    HRESULT STDMETHODCALLTYPE UnlockRequest()
        {
            return S_OK;
        }

        HRESULT STDMETHODCALLTYPE ParseUrl( 
		    /* [in] */ LPCWSTR /*pwzUrl*/,
		    /* [in] */ PARSEACTION /*ParseAction*/,
		    /* [in] */ DWORD /*dwParseFlags*/,
		    /* [out] */ LPWSTR /*pwzResult*/,
		    /* [in] */ DWORD /*cchResult*/,
		    /* [out] */ DWORD* /*pcchResult*/,
		    /* [in] */ DWORD /*dwReserved*/) 
	    {
            //static wchr* blah = L"o3"; 
            //pwzResult = blah;
		    //*pcchResult=strLen(blah)*2+2;
		    
            return INET_E_DEFAULT_ACTION; 
        }

	    virtual HRESULT STDMETHODCALLTYPE CombineUrl( 
		    /* [in] */ LPCWSTR /*pwzBaseUrl*/,
		    /* [in] */ LPCWSTR /*pwzRelativeUrl*/,
		    /* [in] */ DWORD /*dwCombineFlags*/,
		    /* [out] */ LPWSTR /*pwzResult*/,
		    /* [in] */ DWORD /*cchResult*/,
		    /* [out] */ DWORD* /*pcchResult*/,
		    /* [in] */ DWORD /*dwReserved*/)
	    {
		    return E_NOTIMPL;
	    }


	    HRESULT STDMETHODCALLTYPE CompareUrl( 
		    /* [in] */ LPCWSTR /*pwzUrl1*/,
		    /* [in] */ LPCWSTR /*pwzUrl2*/,
		    /* [in] */ DWORD /*dwCompareFlags*/) 
	    {
		    return E_NOTIMPL;
	    }


	    HRESULT STDMETHODCALLTYPE QueryInfo( 
		    /* [in] */ LPCWSTR /*pwzUrl*/,
		    /* [in] */ QUERYOPTION QueryOption,
		    /* [in] */ DWORD /*dwQueryFlags*/,
		    /* [size_is][out][in] */ LPVOID pBuffer,
		    /* [in] */ DWORD cbBuffer,
		    /* [out][in] */ DWORD *pcbBuf,
		    /* [in] */ DWORD /*dwReserved*/) 
	    {
           return INET_E_DEFAULT_ACTION;


	       if(QueryOption == QUERY_IS_SECURE)
	       {
			   if (cbBuffer < sizeof(BOOL)) 
                   return S_FALSE;
			   *((BOOL*)pBuffer)=FALSE;
			   *pcbBuf = sizeof(BOOL);
		       return S_OK;
	       }

		    if (QueryOption == QUERY_USES_CACHE)
            {
			    if (cbBuffer < sizeof(BOOL)) 
                    return S_FALSE;
			    *((BOOL*)pBuffer)=FALSE;
			    *pcbBuf = sizeof(BOOL);
                return S_OK;
            }

            return INET_E_QUERYOPTION_UNKNOWN;
        }
    };
}