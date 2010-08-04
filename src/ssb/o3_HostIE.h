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

#include <exdisp.h>		// IWebBrowser2
#include <Exdispid.h>   // browser events
#include <mshtml.h>		// IHTMLDocument2
#include <mshtmhst.h>	// IDocHostUIHandler

#include "shared/o3_CDropTarget.h"
namespace o3{
    volatile int32_t g_outerComponents = 0;

    int incWrapperCount() {
        return atomicInc(g_outerComponents);
    }

    int decWrapperCount() {
        return atomicDec(g_outerComponents);
    } 

}


namespace o3 {

    #ifdef IEHost_Assert
    #define NOTIMPLEMENTED return(E_NOTIMPL)
    #else
    #define NOTIMPLEMENTED o3_assert(false); return(E_NOTIMPL)
    #endif

    // smart pointers for win32 COM objects
    mscom_ptr(IUnknown);
    mscom_ptr(IWebBrowser2);
    //mscom_ptr(IClassFactory);
    mscom_ptr(IOleObject);
    mscom_ptr(IOleInPlaceFrame);
    mscom_ptr(IOleInPlaceObject);
    mscom_ptr(IOleInPlaceActiveObject);
    mscom_ptr(IOleClientSite);
    mscom_ptr(IDispBridge);    
    mscom_ptr(IDispatch);
    mscom_ptr(IConnectionPointContainer);
    mscom_ptr(IConnectionPoint);
    mscom_ptr(IDropTarget);

    struct HostIE 
        : IOleInPlaceFrame
        , IOleClientSite
        , IOleInPlaceSite
        , IDocHostUIHandler
        , IDispatch
    {

        // helper inner class for handleing the windows messages
        struct cHostProc : cUnk, iWindowProc {
            cHostProc(HostIE* pthis)
                : p_this(pthis)
            {
            }

            virtual ~cHostProc()
            {
            }

            o3_begin_class(cUnk)
                o3_add_iface(iWindowProc)
            o3_end_class()    

            HostIE*  p_this;

            LRESULT CALLBACK wndProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
            {
                // CHECK: think for better solution...
                if (!p_this->m_hwnd)
                    p_this->m_hwnd = hwnd;

	            switch (uMsg)
	            {
		            case WM_SIZE:
			            // Resize the browser object to fit the window    
                        p_this->resizeBrowser(LOWORD(lParam), HIWORD(lParam));
			            return(0);
		            case WM_CREATE:
			            // Embed the browser object into our host window. We need do this only
			            // once. Note that the browser object will start calling some of our
			            // IOleInPlaceFrame and IOleClientSite functions as soon as we start
			            // calling browser object functions in EmbedBrowserObject().
			            if (!p_this->embedBrowser()) 
                            return(-1);

			            // Success
			            return(0);
		            case WM_DESTROY:
			            // Detach the browser object from this window, and free resources.
			            p_this->dropBrowser();

			            // quit this app
                        // TODO: define when closing the browser window will
                        // result in shutting the app down... (multiple windows...)
			            PostQuitMessage(0);

			            return(TRUE);
	            }
	            return(DefWindowProc(hwnd, uMsg, wParam, lParam));
            }
        };

        // helper object to recieve browser events
        struct HostSink : DWebBrowserEvents2                                      
        {
            HostSink(HostIE* pthis)
                :   p_this(pthis)
            {}

            virtual ~HostSink(){}

            mscom_begin_debug(HostSink)
                mscom_add_dinterface(DWebBrowserEvents2)
                mscom_add_interface(IDispatch)
	        mscom_end();   

            ULONG STDMETHODCALLTYPE AddRef() {
                int32_t ret = atomicInc(_m_com.ref_count);
                return (ULONG)ret;
            } 
            ULONG STDMETHODCALLTYPE Release() {              
                int ret = atomicDec(_m_com.ref_count);
                if( ret == 0){ 
                    this->~HostSink(); 
                    g_sys->free(this); 
                } 
                return (ULONG)ret;
            } 	

            HostIE*  p_this;

	        // IDispatch methods
	        STDMETHODIMP GetTypeInfoCount(UINT *pctinfo)
            {
                return E_NOTIMPL;
            }
	        
            STDMETHODIMP GetTypeInfo(UINT iTInfo,LCID lcid,ITypeInfo **ppTInfo)
            {
                return E_NOTIMPL;
            }

	        STDMETHODIMP GetIDsOfNames(REFIID riid,LPOLESTR *rgszNames,
                UINT cNames,LCID lcid,DISPID *rgDispId)
            {
                return E_NOTIMPL;
            }

	        STDMETHODIMP Invoke(DISPID dispIdMember,REFIID riid,LCID lcid,
                WORD wFlags,DISPPARAMS *pDispParams,VARIANT *pVarResult,
                EXCEPINFO *pExcepInfo,UINT *puArgErr)
            {
                // here we can receive the events and play with them
                if (dispIdMember==DISPID_NEWWINDOW2) 
                {
                    // 2 incomming args in reversed order
                    VARIANT& arg1 = pDispParams->rgvarg[1];
                    *arg1.pboolVal  = VARIANT_TRUE;
                    return S_OK;    
                }
                if (dispIdMember==DISPID_NEWWINDOW3) 
                {
                    // 4 incommin args in reverse order...
                    VARIANT& arg1 = pDispParams->rgvarg[3];
                    *arg1.pboolVal  = VARIANT_TRUE;                    
                    p_this->m_web_browser2->Navigate2(&pDispParams->rgvarg[0], 0, 0, 0, 0); 
                    return S_OK;    
                }
                if (dispIdMember==DISPID_BEFORENAVIGATE2)
                {
                    return S_OK;
                }
                return S_OK;
            }
        };

        HostIE(iCtx* ctx)
            : m_hwnd(0)
            , m_ctx(ctx)
            , m_advise_cookie(0)
        {
            m_rect.left = 0;
            m_rect.top = 0;
            m_rect.right = 640;
            m_rect.bottom = 480;
            
            m_wnd_proc = o3_new(cHostProc)(this);
            m_sink = o3_new(HostSink)(this);
            m_drop_target = o3_new(CDropTarget);
            m_o3 = o3_new(cO3)(ctx, 0,0,0);
            Var v = siScr(m_o3);
            ctx->setValue("o3", v);
        }

        virtual ~HostIE()
        {
            dropBrowser();
        }

        mscom_begin_debug(HostIE)
			mscom_add_interface(IOleInPlaceFrame)
			mscom_add_interface(IOleClientSite)
            mscom_add_interface(IOleInPlaceSite)
            mscom_add_interface(IDocHostUIHandler)
            mscom_add_interface(IDispatch)
		mscom_end();

        ULONG STDMETHODCALLTYPE AddRef() {
            int32_t ret = atomicInc(_m_com.ref_count);
            return (ULONG)ret;
        } 
        ULONG STDMETHODCALLTYPE Release() {              
            int ret = atomicDec(_m_com.ref_count);
            if( ret == 0){ 
                this->~HostIE(); 
                g_sys->free(this); 
            } 
            return (ULONG)ret;
        } 	

        // window
        HWND                m_hwnd;
        Str                 m_caption;  // this should be removed...
        RECT                m_rect;     // this should be removed...

        // o3 
        siScr               m_o3;       // o3 object (root)   
        siCtx               m_ctx;      
        siWindowProc           m_wnd_proc; // helper class for handling the win32 messages

        // browser
        SIWebBrowser2		m_web_browser2;     
        
        // protocol
        SIProtocolIE        m_proto_factory;    
        
        // browser events
        SIConnectionPoint   m_connection;       // connection point on the browser for browser events
        SIDispatch          m_sink;             // helper class for receiving browser events
        DWORD               m_advise_cookie;    // cookie for

        // drop target
        SIDropTarget        m_drop_target;

        void displayURL(const char* url)
        {
           if (!m_web_browser2)
               return;

           // create a VARIANT with a BSTR version of the url 
           VARIANT VURL; 
           VariantInit(&VURL);
		   VURL.vt = VT_BSTR;
           Str wrapper(url);
           WStr wurl(wrapper);
           VURL.bstrVal = SysAllocString(wurl.ptr()); 

           //navigation
           m_web_browser2->Navigate2(&VURL, 0, 0, 0, 0); 

           //free resource
           VariantClear(&VURL);
        }

        bool createWindow()
        {
		    WNDCLASSW wc;
            regWndClass(wc);
            
            // Create a window. NOTE: We embed the browser object duing our WM_CREATE handling for
		    // this window.
		    m_hwnd = CreateWindowExW(0, o3_wnd_class_name, L"Is this the caption string?", 
                                WS_OVERLAPPED|WS_CAPTION|WS_SYSMENU|WS_THICKFRAME|WS_MINIMIZEBOX
                                |WS_MAXIMIZEBOX,
							    0, 0, 640, 480,
							    HWND_DESKTOP, NULL, GetModuleHandle(NULL), (LPVOID)m_wnd_proc.ptr());
            siCtx(m_ctx)->setAppWindow(m_hwnd);
            return m_hwnd ? true : false;
        }

        void showWindow()
        {
            ShowWindow(m_hwnd, SW_SHOW);
			UpdateWindow(m_hwnd);
        }

        bool initProtocol()
        {
			m_proto_factory = o3_new(ProtocolIE)(m_ctx);
			ProtocolIE::registerProtocol(m_proto_factory.ptr());

            return true;
        }

        bool embedBrowser()
        {
	        SIClassFactory		class_factory = 0;
	        SIOleObject         browser_object;
	        
        	if ( !CoGetClassObject(CLSID_WebBrowser, 
                    CLSCTX_INPROC_SERVER | CLSCTX_INPROC_HANDLER, 
                    NULL, 
                    IID_IClassFactory, 
                    (void **)&class_factory) 
                && class_factory )
            {
                if (!class_factory->CreateInstance(0, IID_IOleObject, (void**) &browser_object))
                {
                    if (!browser_object->SetClientSite(SIOleClientSite(this).ptr()))
                    {
                        browser_object->SetHostNames(L"O3 Name Of The Host", 0);
                        // Let browser object know that it is embedded in an OLE container.
				        if (!OleSetContainedObject((IUnknown *)browser_object, TRUE) 
                            // set the browser object window size
                            && !browser_object->DoVerb(OLEIVERB_SHOW, NULL, 
                                (IOleClientSite *)this, -1, m_hwnd, &m_rect) 
                            // set m_web_browser2
                            && !browser_object->QueryInterface(
                                IID_IWebBrowser2, (void**)&m_web_browser2))
                        {
                            // set browser display area:
                            m_web_browser2->put_Left(m_rect.left);
                            m_web_browser2->put_Top(m_rect.top);
                            m_web_browser2->put_Width(m_rect.right);
                            m_web_browser2->put_Height(m_rect.bottom);
                            
                            // now for the browser events, we need to fetch a connection point
                            // from the browser and connect our event listener sink object via 
                            // an Advise call
                            SIConnectionPointContainer cpc = m_web_browser2;
                            if (cpc)
                            {
                                // connection point to web browser events
                                cpc->FindConnectionPoint(DIID_DWebBrowserEvents2,&m_connection);                            
                            }
                            if (m_connection)
                            {
                                // registering our listener sink object
                                m_connection->Advise(m_sink,&m_advise_cookie);
                            }
                            return true;
                        }
                    }                    
                    dropBrowser();                    
                }
            }
            return false;
        }

        void dropBrowser()
        {
            if (m_connection)
                m_connection->Unadvise(m_advise_cookie);
            m_advise_cookie = 0;
            m_connection = 0;

            SIOleObject browser_object(m_web_browser2);
            if (browser_object)
                browser_object->Close(OLECLOSE_NOSAVE);
            m_web_browser2 = 0;
            browser_object = 0;
        }

        void resizeBrowser(DWORD width, DWORD height)
        {
	        // Notify the brwoser about the window size change
	        m_web_browser2->put_Width(width);
	        m_web_browser2->put_Height(height);
        }

        void start()
        {
            MSG msg;    
            while (GetMessage(&msg, 0, 0, 0) == 1) 
            {
                if( msg.message == WM_KEYDOWN ) 
                {
                    // tab key for the browser, and other special keys...
                    SIOleInPlaceActiveObject in_plac_ao(m_web_browser2);
                    if (in_plac_ao) 
                    {
                        in_plac_ao->TranslateAccelerator(&msg);                        
                    }                
                }
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }        
        }

// IDocHostUIHandler:
        // display the context menu.
        HRESULT STDMETHODCALLTYPE ShowContextMenu(DWORD dwID, POINT *ppt, IUnknown *pcmdtReserved, IDispatch *pdispReserved)
        {
	        return(S_OK);
        }

        // initi browser UI
        HRESULT STDMETHODCALLTYPE GetHostInfo(DOCHOSTUIINFO *pInfo)
        {
            pInfo->cbSize = sizeof(DOCHOSTUIINFO);

            //set flags here like:
            //DOCHOSTUIFLAG_DIALOG = 0x00000001,
            //DOCHOSTUIFLAG_DISABLE_HELP_MENU = 0x00000002,
            //DOCHOSTUIFLAG_SCROLL_NO = 0x00000008,
            //DOCHOSTUIFLAG_DISABLE_SCRIPT_INACTIVE = 0x00000010,
            //DOCHOSTUIFLAG_OPENNEWWIN = 0x00000020,
            //DOCHOSTUIFLAG_DISABLE_OFFSCREEN = 0x00000040,
            //DOCHOSTUIFLAG_DIV_BLOCKDEFAULT = 0x00000100,
            //DOCHOSTUIFLAG_ACTIVATE_CLIENTHIT_ONLY = 0x00000200,
            //DOCHOSTUIFLAG_URL_ENCODING_DISABLE_UTF8 = 0x00001000,
            //DOCHOSTUIFLAG_URL_ENCODING_ENABLE_UTF8 = 0x00002000,
            //DOCHOSTUIFLAG_ENABLE_FORMS_AUTOCOMPLETE = 0x00004000,
            //DOCHOSTUIFLAG_ENABLE_INPLACE_NAVIGATION = 0x00010000,
            //DOCHOSTUIFLAG_DISABLE_EDIT_NS_FIXUP = 0x00400000,
            //DOCHOSTUIFLAG_LOCAL_MACHINE_ACCESS_CHECK = 0x00800000,
            //DOCHOSTUIFLAG_DISABLE_UNTRUSTEDPROTOCOL = 0x01000000,
            //DOCHOSTUIFLAG_HOST_NAVIGATES = 0x02000000,
            //DOCHOSTUIFLAG_ENABLE_REDIRECT_NOTIFICATION = 0x04000000,
            //DOCHOSTUIFLAG_USE_WINDOWLESS_SELECTCONTROL = 0x08000000,
            //DOCHOSTUIFLAG_USE_WINDOWED_SELECTCONTROL = 0x10000000,
            //DOCHOSTUIFLAG_ENABLE_ACTIVEX_INACTIVATE_MODE = 0x20000000,

	        pInfo->dwFlags = DOCHOSTUIFLAG_NO3DBORDER;

            // double click behaviour
	        pInfo->dwDoubleClick = DOCHOSTUIDBLCLK_DEFAULT;

            return(S_OK);
        }

        HRESULT STDMETHODCALLTYPE ShowUI(DWORD dwID, IOleInPlaceActiveObject *pActiveObject, IOleCommandTarget __RPC_FAR *pCommandTarget, IOleInPlaceFrame __RPC_FAR *pFrame, IOleInPlaceUIWindow *pDoc)
        {
	        return(S_OK);
        }

        HRESULT STDMETHODCALLTYPE HideUI()
        {
	        return(S_OK);
        }

        HRESULT STDMETHODCALLTYPE UpdateUI()
        {
	        return(S_OK);
        }

        HRESULT STDMETHODCALLTYPE EnableModeless(BOOL fEnable)
        {
	        return(S_OK);
        }

        HRESULT STDMETHODCALLTYPE OnDocWindowActivate(BOOL fActivate)
        {
	        return(S_OK);
        }

        HRESULT STDMETHODCALLTYPE OnFrameWindowActivate(BOOL fActivate)
        {
	        return(S_OK);
        }

        HRESULT STDMETHODCALLTYPE ResizeBorder(LPCRECT prcBorder, IOleInPlaceUIWindow *pUIWindow, BOOL fRameWindow)
        {
	        return(S_OK);
        }

        HRESULT STDMETHODCALLTYPE TranslateAccelerator(LPMSG lpMsg, const GUID *pguidCmdGroup, DWORD nCmdID)
        {
            // you can overwrite key strokes here...
	        return(S_FALSE);
        }

        HRESULT STDMETHODCALLTYPE GetOptionKeyPath(LPOLESTR __RPC_FAR *pchKey, DWORD dw)
        {
            // MSDN : "Even if this method is not implemented, the parameter should be set to NULL."
            pchKey = NULL;
	        // default registry settings for now
	        return(S_FALSE);
        }

        // custom drop target
        HRESULT STDMETHODCALLTYPE GetDropTarget(IDropTarget __RPC_FAR *pDropTarget, IDropTarget __RPC_FAR *__RPC_FAR *ppDropTarget)
        {
            if (m_drop_target) {
                *ppDropTarget = m_drop_target;
                m_drop_target->AddRef(); 
                return (S_OK);
            }

            return(S_FALSE);
        }

        // O3 injection to windows.external
        HRESULT STDMETHODCALLTYPE GetExternal(IDispatch **ppDispatch)
        {
            
            *ppDispatch = SIDispatch(this).ptr();
             
             this->AddRef(); // shall I addref here?
             return(S_OK);
        }   

        // URL modification
        HRESULT STDMETHODCALLTYPE TranslateUrl(DWORD dwTranslate, OLECHAR *pchURLIn, OLECHAR **ppchURLOut)
        {            
	        *ppchURLOut = 0;
            return(S_FALSE);
        }

        // if we ever want to disable / customize cut and paste, it can be done here...
        HRESULT STDMETHODCALLTYPE FilterDataObject(IDataObject *pDO, IDataObject **ppDORet)
        {            
	        *ppDORet = 0;
	        return(S_FALSE);
        }


// IOleClientSite:


        HRESULT STDMETHODCALLTYPE SaveObject()
        {
	        NOTIMPLEMENTED;
        }

        HRESULT STDMETHODCALLTYPE GetMoniker(DWORD dwAssign, DWORD dwWhichMoniker, IMoniker **ppmk)
        {
	        NOTIMPLEMENTED;
        }

        HRESULT STDMETHODCALLTYPE GetContainer(LPOLECONTAINER *ppContainer)
        {
	        // no container support for now
	        *ppContainer = 0;

	        return(E_NOINTERFACE);
        }

        HRESULT STDMETHODCALLTYPE ShowObject()
        {
	        return(NOERROR);
        }

        HRESULT STDMETHODCALLTYPE OnShowWindow(BOOL fShow)
        {
	        NOTIMPLEMENTED;
        }

        HRESULT STDMETHODCALLTYPE RequestNewObjectLayout()
        {
	        NOTIMPLEMENTED;
        }

// IOleInPlaceSite


        HRESULT STDMETHODCALLTYPE ContextSensitiveHelp(BOOL fEnterMode)
        {
	        NOTIMPLEMENTED;
        }

        HRESULT STDMETHODCALLTYPE CanInPlaceActivate()
        {
	        return(S_OK);
        }

        HRESULT STDMETHODCALLTYPE OnInPlaceActivate()
        {
	        return(S_OK);
        }

        HRESULT STDMETHODCALLTYPE OnUIActivate()
        {
	        return(S_OK);
        }

        HRESULT STDMETHODCALLTYPE GetWindowContext(IOleInPlaceFrame **lplpFrame, IOleInPlaceUIWindow  **lplpDoc, LPRECT lprcPosRect, LPRECT lprcClipRect, LPOLEINPLACEFRAMEINFO lpFrameInfo)
        {

	        *lplpFrame = SIOleInPlaceFrame(this).ptr();

	        // no OLEINPLACEUIWINDOW
	        *lplpDoc = 0;

	        lpFrameInfo->fMDIApp = FALSE;
	        lpFrameInfo->hwndFrame = m_hwnd;
	        lpFrameInfo->haccel = 0;
	        lpFrameInfo->cAccelEntries = 0;

            // set the draw area
        	GetClientRect(m_hwnd, lprcPosRect);
        	GetClientRect(m_hwnd, lprcClipRect);

	        return(S_OK);
        }

        HRESULT STDMETHODCALLTYPE Scroll(SIZE scrollExtent)
        {
	        NOTIMPLEMENTED;
        }

        HRESULT STDMETHODCALLTYPE OnUIDeactivate(BOOL fUndoable)
        {
	        return(S_OK);
        }

        HRESULT STDMETHODCALLTYPE OnInPlaceDeactivate()
        {
	        return(S_OK);
        }

        HRESULT STDMETHODCALLTYPE DiscardUndoState()
        {
	        NOTIMPLEMENTED;
        }

        HRESULT STDMETHODCALLTYPE DeactivateAndUndo()
        {
	        NOTIMPLEMENTED;
        }

        HRESULT STDMETHODCALLTYPE OnPosRectChange(LPCRECT lprcPosRect)
        {
            SIOleInPlaceObject	inplace(m_web_browser2);
	        if (inplace)
	        {
		        // rect where it can draw.
		        inplace->SetObjectRects(lprcPosRect, lprcPosRect);
	        }

	        return(S_OK);
        }


// IOleInPlaceFrame


        HRESULT STDMETHODCALLTYPE GetWindow(HWND *lphwnd)
        {
	        *lphwnd = m_hwnd;
	        return(S_OK);
        }

        HRESULT STDMETHODCALLTYPE GetBorder(LPRECT lprectBorder)
        {
	        NOTIMPLEMENTED;
        }

        HRESULT STDMETHODCALLTYPE RequestBorderSpace(LPCBORDERWIDTHS pborderwidths)
        {
	        NOTIMPLEMENTED;
        }

        HRESULT STDMETHODCALLTYPE SetBorderSpace(LPCBORDERWIDTHS pborderwidths)
        {
	        NOTIMPLEMENTED;
        }

        HRESULT STDMETHODCALLTYPE SetActiveObject(IOleInPlaceActiveObject *pActiveObject, LPCOLESTR pszObjName)
        {
	        return(S_OK);
        }

        HRESULT STDMETHODCALLTYPE InsertMenus(HMENU hmenuShared, LPOLEMENUGROUPWIDTHS lpMenuWidths)
        {
	        NOTIMPLEMENTED;
        }

        HRESULT STDMETHODCALLTYPE SetMenu(HMENU hmenuShared, HOLEMENU holemenu, HWND hwndActiveObject)
        {
	        return(S_OK);
        }

        HRESULT STDMETHODCALLTYPE RemoveMenus(HMENU hmenuShared)
        {
	        NOTIMPLEMENTED;
        }

        HRESULT STDMETHODCALLTYPE SetStatusText(LPCOLESTR pszStatusText)
        {
	        return(S_OK);
        }

        HRESULT STDMETHODCALLTYPE TranslateAccelerator(LPMSG lpmsg, WORD wID)
        {
	        NOTIMPLEMENTED;
        }

// IDispatch

        HRESULT STDMETHODCALLTYPE GetTypeInfoCount(unsigned int *pctinfo)
        {
	        return(E_NOTIMPL);
        }

        HRESULT STDMETHODCALLTYPE GetTypeInfo(unsigned int iTInfo, LCID lcid, ITypeInfo **ppTInfo)
        {
	        return(E_NOTIMPL);
        }

        HRESULT STDMETHODCALLTYPE GetIDsOfNames(REFIID riid, OLECHAR ** rgszNames, unsigned int cNames, LCID lcid, DISPID * rgDispId)
        {
            HRESULT ret = S_OK;
            for (unsigned int i=0; i<cNames; i++) {
                if (strEquals(rgszNames[i], L"o3"))
                    rgDispId[i] = 1;
                else {
                    rgDispId[i] = -1;
                    ret = DISP_E_UNKNOWNNAME;
                }
            }

	        return(ret);
        }

        HRESULT STDMETHODCALLTYPE Invoke(DISPID dispIdMember, REFIID riid, LCID lcid, WORD wFlags, 
            DISPPARAMS * pDispParams, VARIANT *pVarResult, EXCEPINFO *pExcepInfo, unsigned int *puArgErr)
        {
            if ( 1 == dispIdMember  
                && DISPATCH_PROPERTYGET == wFlags
                && pVarResult ) 
            {
                Var ret = m_o3;
                Var_to_VARIANT(ret, *pVarResult, siCtx(m_ctx));
                return (S_OK);
            }

	        return(DISP_E_MEMBERNOTFOUND);
        }


    };

}