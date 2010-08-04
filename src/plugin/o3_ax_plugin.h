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

#ifdef O3_WIN32
//#include "Psapi.h"
//#include <exdisp.h>		// IWebBrowser2
#include <Exdispid.h>
#include <SHLGUID.h>
#include <mshtml.h>
#include <mshtmdid.h>
#include <olectl.h>
#include "shared/o3_glue_idispatch.h"
#include "protocol/o3_protocol.h"

DEFINE_GUID(IID_IJAxCtrl, 0xddbbe8d1, 0x8ee4, 0x4037, 0x81, 
    0x6c, 0x42, 0xd1, 0x5, 0xd6, 0x9f, 0xf4);

GUID CLSID_IJAxCtrl ={0xAAAAAAAA,0x1111,0xBBBB,{0x11, 0x11, 
    0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC}};

wchar_t stemGUID[] = O3_PLUGIN_GUID;
wchar_t baseAppName[] = O3_APP_NAME;


namespace o3 {
    volatile int g_win32_outstanding_objects = 0;
    volatile int g_win32_lock_count = 0;
    extern int g_outerComponents;
    HINSTANCE    g_win32_hinst = NULL;

    mscom_ptr(IDispBridge);
	mscom_ptr(IConnectionPoint);
	mscom_ptr(IConnectionPointContainer);
	mscom_ptr(IDispatch);
	mscom_ptr(IHTMLDocument2);
	mscom_ptr(IWebBrowser2);
	mscom_ptr(IHTMLWindow2);
	mscom_ptr(IHTMLEventObj);
	mscom_ptr(IHTMLEventObj2);
	mscom_ptr(IHTMLElement);
	mscom_ptr(IDispatchEx);
	mscom_ptr_d(HTMLDocumentEvents);	

    int incWrapperCount() {
        int ret = atomicInc(g_win32_outstanding_objects);
        return ret;
    }

    int decWrapperCount() {
		int ret = atomicDec(g_win32_outstanding_objects);
        return ret;
    }


	struct IJAxCtrl :
		public IDispatchEx,
		public IPersistStreamInit,
		public IOleControl,
		public IOleObject,
		public IOleInPlaceActiveObject,
		public IViewObjectEx,
		public IQuickActivate,
		public IOleInPlaceObjectWindowless,
		public IObjectSafety,
		public IPersistPropertyBag
	{
		virtual ULONG STDMETHODCALLTYPE AddRef() = 0;
		virtual ULONG STDMETHODCALLTYPE Release() = 0;
	};


    struct cCtx1 : cCtx, iCtx1 
    {
		cCtx1() :
			m_track(0)
        {
        
        }

	    o3_begin_class(cCtx)
		    o3_add_iface(iCtx1)
	    o3_end_class()

        ComTrack*       m_track;
		HWND			m_hwnd;

        virtual ComTrack** track()
        {
           return &m_track;
        }

        virtual void tear() 
        {
            for (ComTrack *i = m_track, *j = 0; i; i = j) {
                j = i->m_next;
                i->m_phead = 0;
                i->tear();
            }
        }

    };


	BOOL CALLBACK FindIEServerWindow(HWND hwnd, LPARAM lParam);

    typedef BOOL (STDMETHODCALLTYPE* bfpointerdw)(ULONG_PTR);

    class CJAxCtrl : public IJAxCtrl
    {

/****************************/
/* CJAxCtrl implementation:	*/
/****************************/
    public:
		CJAxCtrl() 
		{               
			m_blocked = false;	
			m_ctx = o3_new(cCtx1)();			
			siScr root = o3_new(cO3)(m_ctx, 0,0,0);          

            m_bridge = o3_new(CDispBridge)(siCtx1(m_ctx),root) ;
			m_proto_factory = o3_new(ProtocolIE)(m_ctx);
			ProtocolIE::registerProtocol(m_proto_factory.ptr());
        }
        
        virtual ~CJAxCtrl()
		{          
			if (m_proto_factory) {
				ProtocolIE::unregisterProtocol(m_proto_factory.ptr());
				m_proto_factory = 0;
			}

			if(m_bridge) {
				if (m_ctx)
					siCtx1(m_ctx)->tear();
				
				m_bridge = 0;                
                m_ctx = 0;
            }
        }

        mscom_begin_debug(CJAxCtrl)
            mscom_add_interface(IJAxCtrl)
            mscom_add_interface(IPersistStreamInit)
            mscom_add_interface(IOleControl)
            mscom_add_interface(IOleObject)
            mscom_add_interface(IOleInPlaceActiveObject)
            mscom_add_interface(IViewObjectEx)
            mscom_add_interface(IViewObject2)
            mscom_add_interface(IViewObject)
            mscom_add_iid_iface(IOleWindow,IOleInPlaceActiveObject)            
            mscom_add_interface(IOleInPlaceObject)
            mscom_add_interface(IQuickActivate)
            mscom_add_interface(IOleInPlaceObjectWindowless)
            mscom_add_interface(IObjectSafety)
            mscom_add_interface(IPersistPropertyBag)
            mscom_add_interface(IDispatchEx)
            mscom_add_interface(IDispatch)
        mscom_end();

        ULONG STDMETHODCALLTYPE AddRef() {
            int32_t ret = atomicInc(_m_com.ref_count);
            return (ULONG)ret;
        } 
        ULONG STDMETHODCALLTYPE Release() {              
            int ret = atomicDec(_m_com.ref_count);
            if( ret == 0){ 
                this->~CJAxCtrl(); 
                g_sys->free(this); 
            } 
            return (ULONG)ret;
        } 	

        siCtx                       m_ctx;
        SIDispBridge                m_bridge;
        HiddenWindow                m_hidden_window;
		SIProtocolIE				m_proto_factory;
		SIWebBrowser2				m_webbrowser;
		bool						m_blocked;


        // IDispatch
        HRESULT STDMETHODCALLTYPE CJAxCtrl::GetTypeInfoCount(unsigned int FAR*  pctinfo){
            *pctinfo = 0;
            return(S_OK);
        }

        HRESULT STDMETHODCALLTYPE CJAxCtrl::GetTypeInfo(unsigned int  iTInfo, LCID  lcid, ITypeInfo FAR* FAR*  ppTInfo ){ return E_NOTIMPL; }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::GetIDsOfNames(REFIID  riid, OLECHAR FAR* FAR*  rgszNames, 
            unsigned int  cNames, LCID   lcid, DISPID FAR*  rgDispId ){ return E_NOTIMPL; }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::Invoke(DISPID  dispIdMember, REFIID  riid, LCID  lcid, WORD  wFlags,
            DISPPARAMS FAR*  pDispParams, VARIANT FAR*  pVarResult, EXCEPINFO FAR*  pExcepInfo, unsigned int FAR*  puArgErr){ return E_NOTIMPL; }


        //IDispatchEx
        HRESULT STDMETHODCALLTYPE DeleteMemberByDispID(DISPID id){
            if (m_blocked)
				return E_NOTIMPL;
			else
				return m_bridge->DeleteMemberByDispID(id);
        }

        HRESULT STDMETHODCALLTYPE DeleteMemberByName(BSTR bstrName, DWORD grfdex){
			if (m_blocked)
				return E_NOTIMPL;
			else
				return m_bridge->DeleteMemberByName(bstrName, grfdex);
        }

        HRESULT STDMETHODCALLTYPE GetDispID(BSTR bstrName, DWORD grfdex, DISPID *pid){
			if (m_blocked)
				return E_NOTIMPL;
			else            
				return m_bridge->GetDispID( bstrName, grfdex, pid );
        }

        HRESULT STDMETHODCALLTYPE GetMemberName(DISPID id, BSTR *pbstrName){
			if (m_blocked)
				return E_NOTIMPL;
			else
				return m_bridge->GetMemberName( id, pbstrName );
        }
     
        HRESULT STDMETHODCALLTYPE GetMemberProperties(DISPID id, DWORD grfdexFetch, DWORD *pgrfdex){
			if (m_blocked)
				return E_NOTIMPL;
			else
				return m_bridge->GetMemberProperties( id, grfdexFetch, pgrfdex );
        }
        
        HRESULT STDMETHODCALLTYPE GetNameSpaceParent(IUnknown **ppunk){
			if (m_blocked)
				return E_NOTIMPL;
			else
				return m_bridge->GetNameSpaceParent( ppunk );
        }

        HRESULT STDMETHODCALLTYPE GetNextDispID(DWORD grfdex, DISPID id, DISPID *pid){
			if (m_blocked)
				return E_NOTIMPL;
			else
				return m_bridge->GetNextDispID(grfdex, id, pid);
        }

        HRESULT STDMETHODCALLTYPE InvokeEx(DISPID id, LCID lcid, WORD wFlags, DISPPARAMS *pdp, VARIANT *pVarRes, EXCEPINFO *pei, IServiceProvider *pspCaller){
			if (m_blocked)
				return E_NOTIMPL;
			else
				return m_bridge->InvokeEx( id, lcid, wFlags, pdp, pVarRes, pei, pspCaller);
        }

        //IPointerInactive
        HRESULT STDMETHODCALLTYPE GetActivationPolicy( DWORD* pdwPolicy ){
            //*pdwPolicy = POINTERINACTIVE_ACTIVATEONENTRY;
            return E_NOTIMPL;
        }

        HRESULT STDMETHODCALLTYPE OnInactiveMouseMove( LPCRECT pRectBounds, LONG x, LONG y, DWORD grfKeyState ){ return E_NOTIMPL; }
        HRESULT STDMETHODCALLTYPE OnInactiveSetCursor( LPCRECT pRectBounds, LONG x, LONG y, DWORD dwMouseMsg, BOOL fSetAlways ){ return E_NOTIMPL; }
        

        // IPersist
        HRESULT STDMETHODCALLTYPE CJAxCtrl::GetClassID( CLSID * pClassID){
            *pClassID = CLSID_IJAxCtrl;
            return(S_OK);
        }

        //IPersistStreamInit
        HRESULT STDMETHODCALLTYPE CJAxCtrl::IsDirty(void){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::Load(LPSTREAM pStm){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::Save(LPSTREAM pStm, BOOL fClearDirty){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::GetSizeMax(ULARGE_INTEGER * pcbSize){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::InitNew(void){ return(E_NOTIMPL); }

        //IOleControl
        HRESULT STDMETHODCALLTYPE CJAxCtrl::GetControlInfo(CONTROLINFO* pCI){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::OnMnemonic(LPMSG pMsg){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::OnAmbientPropertyChange(DISPID dispID){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::FreezeEvents(BOOL bFreeze){ return(E_NOTIMPL); }

        //IOleObject
        HRESULT STDMETHODCALLTYPE CJAxCtrl::SetClientSite(IOleClientSite *pClientSite){
            if (!pClientSite) 
                return (S_OK);

            HRESULT hret(0);                       
            IServiceProvider *srvprov = 0, *srvprov2 = 0;
			IConnectionPointContainer*  pConnectionPointContainer = 0;
            
            hret = pClientSite->QueryInterface(IID_IServiceProvider,
				(void**)(&srvprov));

            if (hret == S_OK && srvprov){
                hret = srvprov->QueryService(SID_STopLevelBrowser,
					IID_IServiceProvider, (void **)(&srvprov2));
            }

            if (hret == S_OK && srvprov2){
                hret = srvprov2->QueryService(SID_SWebBrowserApp, 
					IID_IWebBrowser2, (void **)(&m_webbrowser));
            }
			
			HRESULT ret = S_OK;
			if (m_webbrowser) {
				BSTR url;
				m_webbrowser->get_LocationURL(&url);
				Str url2 = Str(WStr(url));
				m_ctx->mgr()->setCurrentUrl(url2);
				Str host = hostFromURL(url2);
				if (strCaseEquals(host.ptr(),"localhost")
					 || strCaseEquals(host.ptr(),"www.ajax.org"))
				{
					m_blocked = false;
				} 
				else 
				{
					// we don't want to expose dangerous components,
					// so o3 is only allowed to be used from safe places for now										
					m_blocked = true;
					ret = E_UNEXPECTED;
				}

				SysFreeString(url);

				// getting a handle to the window that is responsible for showing
				// the current site, the coordinates of that window are needed
				HWND hwnd;
				if (S_OK == m_webbrowser->get_HWND((SHANDLE_PTR*)&hwnd)) {
					EnumChildWindows(hwnd, FindIEServerWindow, (LPARAM)this );				
				}
			}

			if (pConnectionPointContainer) srvprov->Release();
            if (srvprov) srvprov->Release();
            if (srvprov2) srvprov2->Release();        
            return(ret);
        }




        HRESULT STDMETHODCALLTYPE CJAxCtrl::GetClientSite(IOleClientSite **ppClientSite){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::SetHostNames(LPCOLESTR szContainerApp,LPCOLESTR szContainerObj){ return(E_NOTIMPL); }

        HRESULT STDMETHODCALLTYPE CJAxCtrl::Close(DWORD dwSaveOption){
			
			if (m_proto_factory) {
				ProtocolIE::unregisterProtocol(m_proto_factory.ptr());
				m_proto_factory = 0;
			}

            if (m_ctx)
                siCtx1(m_ctx)->tear();
                           
            if (m_bridge)
                m_bridge = 0;

            m_ctx = 0;
            return(S_OK);
        }

        HRESULT STDMETHODCALLTYPE CJAxCtrl::SetMoniker(DWORD dwWhichMoniker,IMoniker *pmk){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::GetMoniker(DWORD dwAssign,DWORD dwWhichMoniker,IMoniker **ppmk){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::InitFromData(IDataObject *pDataObject,BOOL fCreation,DWORD dwReserved){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::GetClipboardData(DWORD dwReserved,IDataObject **ppDataObject){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::DoVerb(LONG iVerb,LPMSG lpmsg,IOleClientSite *pActiveSite,LONG lindex,
            HWND hwndParent,LPCRECT lprcPosRect){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::EnumVerbs(IEnumOLEVERB **ppEnumOleVerb){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::Update(){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::IsUpToDate(){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::GetUserClassID(CLSID *pClsid){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::GetUserType(DWORD dwFormOfType,LPOLESTR *pszUserType){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::SetExtent(DWORD dwDrawAspect,SIZEL  *psizel){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::GetExtent(DWORD dwDrawAspect,SIZEL *psizel){ return E_NOTIMPL; }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::Advise(IAdviseSink *pAdvSink,DWORD *pdwConnection){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::Unadvise(DWORD dwConnection){ return(E_NOTIMPL); }

        HRESULT STDMETHODCALLTYPE CJAxCtrl::EnumAdvise(IEnumSTATDATA **ppenumAdvise){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::GetMiscStatus(DWORD dwAspect,DWORD *pdwStatus){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::SetColorScheme(LOGPALETTE *pLogpal){ return(E_NOTIMPL); }


        //IViewObject
        HRESULT STDMETHODCALLTYPE CJAxCtrl::Draw(DWORD dwAspect,LONG lindex,void* pvAspect,DVTARGETDEVICE* ptd,HDC hicTargetDev,
            HDC hdcDraw, LPCRECTL lprcBounds, LPCRECTL lprcWBounds,bfpointerdw pfnContinue,ULONG_PTR dwContinue){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::GetColorSet(DWORD dwAspect, LONG lindex, void * pvAspect, DVTARGETDEVICE * ptd,
            HDC hicTargetDev, LOGPALETTE ** ppColorSet){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::Freeze(DWORD dwAspect,LONG lindex,void* pvAspect,DWORD* pdwFreeze){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::GetAdvise(DWORD* pdwAspect,DWORD* padvf,IAdviseSink** ppAdvSink){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::SetAdvise(DWORD dwAspect,DWORD advf,IAdviseSink* pAdvSink){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::Unfreeze(DWORD dwFreeze){ return(E_NOTIMPL); }

        //IViewObject2
        HRESULT STDMETHODCALLTYPE CJAxCtrl::GetExtent(DWORD dwAspect,LONG lindex,DVTARGETDEVICE* ptd,LPSIZEL lpsizel){ return E_NOTIMPL; }

        //IViewObjectEx
        HRESULT STDMETHODCALLTYPE CJAxCtrl::GetNaturalExtent(DWORD dwAspect,LONG lindex,DVTARGETDEVICE* ptd,HDC hicTargetDev,
            DVEXTENTINFO* pExtentInfo,LPSIZEL pSizel){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::GetRect(DWORD dwAspect,LPRECTL pRect){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::GetViewStatus(DWORD* pdwStatus){ return(E_NOTIMPL); }

        HRESULT STDMETHODCALLTYPE CJAxCtrl::QueryHitPoint(DWORD dwAspect,LPCRECT pRectBounds,POINT ptlLoc,LONG lCloseHint,DWORD* pHitResult){
            //*pHitResult = HITRESULT_HIT;
            return(E_NOTIMPL);
        }

        HRESULT STDMETHODCALLTYPE CJAxCtrl::QueryHitRect(DWORD dwAspect,LPCRECT pRectBounds,LPCRECT pRectLoc,
            LONG lCloseHint,DWORD* pHitResult){ return(E_NOTIMPL); }

        //IQuickActivate
        HRESULT STDMETHODCALLTYPE CJAxCtrl::QuickActivate(QACONTAINER* pQaContainer,QACONTROL* pQaControl){        
            SetClientSite(pQaContainer->pClientSite);
            return(S_OK);
        }

        HRESULT STDMETHODCALLTYPE CJAxCtrl::SetContentExtent(LPSIZEL psizel){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::GetContentExtent(LPSIZEL psizel){ return(E_NOTIMPL); }

        //IOleWindow
        HRESULT STDMETHODCALLTYPE CJAxCtrl::GetWindow(HWND * phwnd){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::ContextSensitiveHelp(BOOL fEnterMode){ return(E_NOTIMPL); }

        //IOleInPlaceActiveObject
        HRESULT STDMETHODCALLTYPE CJAxCtrl::TranslateAccelerator(LPMSG lpmsg){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::OnFrameWindowActivate(BOOL fActivate){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::OnDocWindowActivate(BOOL fActivate){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::ResizeBorder(LPCRECT prcBorder,IOleInPlaceUIWindow* pUIWindow,BOOL fFrameWindow){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::EnableModeless(BOOL fEnable){ return(E_NOTIMPL); }

        //IOleInPlaceObject
        HRESULT STDMETHODCALLTYPE CJAxCtrl::InPlaceDeactivate(){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::UIDeactivate(){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::SetObjectRects(LPCRECT lprcPosRect,LPCRECT lprcClipRect){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::ReactivateAndUndo(){ return(E_NOTIMPL); }

        //IOleInPlaceObjectWindowless
        HRESULT STDMETHODCALLTYPE CJAxCtrl::OnWindowMessage(UINT msg,WPARAM wParam,LPARAM lParam,LRESULT* plResult){ return(E_NOTIMPL);}
        HRESULT STDMETHODCALLTYPE CJAxCtrl::GetDropTarget(IDropTarget** ppDropTarget){ return(E_NOTIMPL); }

        //IObjectSafety
        HRESULT STDMETHODCALLTYPE CJAxCtrl::GetInterfaceSafetyOptions(REFIID riid,DWORD *pdwSupportedOptions,DWORD *pdwEnabledOptions){
            *pdwSupportedOptions = INTERFACESAFE_FOR_UNTRUSTED_DATA|INTERFACESAFE_FOR_UNTRUSTED_CALLER;
            *pdwEnabledOptions   = INTERFACESAFE_FOR_UNTRUSTED_DATA|INTERFACESAFE_FOR_UNTRUSTED_CALLER;
            return(S_OK);
        }

        HRESULT STDMETHODCALLTYPE CJAxCtrl::SetInterfaceSafetyOptions(REFIID riid,DWORD dwOptionSetMask,DWORD dwEnabledOptions){
            if (riid == IID_IDispatchEx || riid == IID_IDispatch || riid == IID_IPersistPropertyBag)
                return(S_OK);//E_NOTIMPL);
            else
                return (E_NOINTERFACE);
        }

        //IPersistPropertyBag
        //HRESULT STDMETHODCALLTYPE CJAxCtrl::InitNew(VOID){}
        HRESULT STDMETHODCALLTYPE CJAxCtrl::Load(IPropertyBag *pPropBag,IErrorLog *pErrorLog){ return(E_NOTIMPL); }
        HRESULT STDMETHODCALLTYPE CJAxCtrl::Save(IPropertyBag *pPropBag,BOOL fClearDirty,BOOL fSaveAllProperties){ return(E_NOTIMPL); }

    };

	BOOL CALLBACK FindIEServerWindow(HWND hwnd, LPARAM lParam) 
	{
		CJAxCtrl* pthis = (CJAxCtrl*)lParam;
		WStr clsName; clsName.reserve(256);
		int s = GetClassNameW(hwnd, clsName.ptr(), 256);
		clsName.resize( size_t (s>0 ? s : 0));
		if (strEquals(L"Internet Explorer_Server",clsName.ptr())) 
		{
			siCtx ctx(pthis->m_ctx);
			ctx->setAppWindow(hwnd);
			return FALSE;
		}
		return TRUE;
	}


    wchar_t tmpModuleFile [MAX_PATH];

/****************************/
/* COM class factory:		*/
/****************************/

    class CJAxCtrlClassFactory : public IClassFactory
    {
        public:
        // The IClassFactory object ///////////////////////////////////////////////////////

        CJAxCtrlClassFactory() {
        }

        virtual ~CJAxCtrlClassFactory() {
        }

        mscom_begin_debug(CJAxCtrlClassFactory)
            mscom_add_interface(IClassFactory)
        mscom_end();

        ULONG STDMETHODCALLTYPE AddRef() {
            int32_t ret = atomicInc(_m_com.ref_count);
            return (ULONG)ret;
        } 
        ULONG STDMETHODCALLTYPE Release() {              
            int ret = atomicDec(_m_com.ref_count);
            if( ret == 0){ 
                this->~CJAxCtrlClassFactory(); 
                g_sys->free(this); 
            } 
            return (ULONG)ret;
        } 

        HRESULT STDMETHODCALLTYPE CreateInstance(IUnknown* pUnkOuter,
                                                                 REFIID    riid,
                                                                 void**    ppv) {    
        HRESULT hrRet;
        CJAxCtrl* pJAxCtrl;

            if ( NULL != pUnkOuter )
                return CLASS_E_NOAGGREGATION;

            *ppv = NULL;

            pJAxCtrl = (CJAxCtrl*) o3_new( CJAxCtrl );

            if ( NULL == pJAxCtrl )
                return E_OUTOFMEMORY;

            pJAxCtrl->AddRef();
            hrRet = pJAxCtrl->QueryInterface ( riid, ppv );
            pJAxCtrl->Release();

            return hrRet;
        }

        HRESULT STDMETHODCALLTYPE LockServer(BOOL flock) {
            if (flock) atomicInc(g_win32_lock_count);
            else atomicDec(g_win32_lock_count);

            return(NOERROR);
        }

    };
}

/****************************/
/* Dll functions:			*/
/****************************/

STDAPI DllCanUnloadNow(void) {
    using namespace o3;
    HRESULT ret = (!g_win32_outstanding_objects && !g_win32_lock_count) ? S_OK : S_FALSE;
    return ret;
}

STDAPI DllGetClassObject(REFCLSID objGuid, REFIID factoryGuid, void **factoryHandle) {
    using namespace o3;
    register HRESULT        hr;
    if (IsEqualCLSID(objGuid, CLSID_IJAxCtrl)) {
        CJAxCtrlClassFactory *fac = 
            (CJAxCtrlClassFactory*) o3_new( CJAxCtrlClassFactory) ();
        fac->AddRef();
        hr = fac->QueryInterface(factoryGuid, factoryHandle);
        fac->Release();
    }
    else
    {
        *factoryHandle = 0;
        hr = CLASS_E_CLASSNOTAVAILABLE;
    }

    return(hr);
}



STDAPI DllRegisterServerCust(bool all_usr, wchar_t* path) {
    using namespace o3;
    HKEY base_reg = all_usr ? HKEY_LOCAL_MACHINE : HKEY_CURRENT_USER ;
    #define __retiferror(X) if( ERROR_SUCCESS != X ) goto reg_error;
    #define __closekey(X) if( X ) RegCloseKey( X ); X = NULL

    HKEY  hCLSIDKey(0), hInProcSvrKey(0), hKey(0), hKey2(0), hIIDKey(0), hAPPIDKey(0);

    LONG  lRet;
    wchar_t modulePath [MAX_PATH];

    WStr classes = L"software\\Classes\\";
    WStr classDesc, threadingModel, progId, progVIId;
    WStr clsid, appid, iid, appGUID;

    appGUID.appendf(L"%s%s%s", L"{", stemGUID, L"}");
    classDesc.appendf(L"%s%s%s", baseAppName, L"-", stemGUID);
    threadingModel.appendf( L"%s", L"Apartment");
    progId.appendf(L"%s%s%s%s%s", classes.ptr(), baseAppName, L"-", stemGUID, L".1");
    progVIId.appendf(L"%s%s%s%s", classes.ptr(), baseAppName, L"-", stemGUID);
    clsid.appendf(L"%s%s%s", classes.ptr(), L"CLSID\\", appGUID.ptr());
    iid.appendf(L"%s%s%s", classes.ptr(), L"Interface\\",  
        L"{DDBBE8D1-8EE4-4037-816C-42D105D69FF4}");   
    appid.appendf(L"%s%s%s", classes.ptr(), L"APPID\\", appGUID.ptr());    


    // Create a key under CLSID for our COM server.
    lRet = regCreate(  base_reg, clsid, hCLSIDKey); 
        __retiferror(lRet);

    // The default value of the key is a human-readable description of the coclass.
    lRet = regSet( hCLSIDKey, NULL, classDesc ); 
        __retiferror(lRet);

    // Create the InProcServer32 key, which holds info about our coclass.
    lRet = regCreate( hCLSIDKey, L"InProcServer32", hInProcSvrKey); 
        __retiferror(lRet);

    // The default value of the InProcServer32 key holds the full path to our DLL.
    if (path==0)
        GetModuleFileNameW ( g_win32_hinst, modulePath, MAX_PATH );

    lRet = regSet( hInProcSvrKey, NULL, path ? path : modulePath ); 
        __retiferror(lRet);
    lRet = regSet( hInProcSvrKey, L"ThreadingModel", threadingModel ); 
        __retiferror(lRet); __closekey(hInProcSvrKey);

    // Some more registry settings
    lRet = regCreate( hCLSIDKey, L"control", hKey);
        __retiferror(lRet); __closekey(hKey);

    lRet = regCreate( hCLSIDKey, L"Implemented Categories", hKey);
        __retiferror(lRet);
    lRet = regCreate( hKey, L"{7DD95801-9882-11CF-9FA9-00AA006C42C4}", hKey2 );
        __retiferror(lRet);    __closekey(hKey2);
    lRet = regCreate( hKey, L"{7DD95802-9882-11CF-9FA9-00AA006C42C4}", hKey2);
        __retiferror(lRet);    __closekey(hKey); __closekey(hKey2);
    lRet = regCreate( hCLSIDKey, L"ProgID", hKey);
        __retiferror(lRet);
    lRet = regSet( hKey, NULL, progId );
        __retiferror(lRet); __closekey(hKey);
    lRet = regCreate( hCLSIDKey, L"TypeLib", hKey);
        __retiferror(lRet);
    lRet = regCreate( hCLSIDKey, L"Version", hKey);
        __retiferror(lRet);
    lRet = regCreate( hCLSIDKey, L"VersionIndependentProgID", hKey);
        __retiferror(lRet);
    lRet = regSet( hKey, NULL, progVIId );
        __retiferror(lRet); __closekey(hKey);

    //Interface registration
    lRet = regCreate( base_reg, iid, hIIDKey);
        __retiferror(lRet);
    lRet = regCreate( hIIDKey, L"ProxyStubClsid", hKey);
        __retiferror(lRet);
    lRet = regSet( hKey, NULL, L"{00020424-0000-0000-C000-000000000046}" );
        __retiferror(lRet); __closekey(hKey);
    lRet = regCreate ( hIIDKey, L"ProxyStubClsid32", hKey);
        __retiferror(lRet);
    lRet = regSet( hKey, NULL, L"{00020424-0000-0000-C000-000000000046}" );
        __retiferror(lRet); __closekey(hKey);
    lRet = regCreate( hIIDKey, L"TypeLib", hKey);
        __retiferror(lRet);
    lRet = regSet( hKey, L"Version",L"1.0" );
        __retiferror(lRet); __closekey(hKey);
  
    //App reg
    lRet = regCreate(  base_reg, progId, hKey );
        __retiferror(lRet);
    lRet = regCreate (  hKey, L"CLSID", hKey2 );
    lRet = regSet( hKey2, NULL, appGUID );
        __retiferror(lRet); __closekey(hKey); __closekey(hKey2);
    lRet = regCreate(  base_reg, progVIId, hKey );
        __retiferror(lRet);
    lRet = regCreate(  hKey, L"CLSID", hKey2 );
    lRet = regSet( hKey2, NULL, appGUID );
        __retiferror(lRet); __closekey(hKey2);
    lRet = regCreate ( hKey, L"CurVersion", hKey2 );
    lRet = regSet( hKey2, NULL, progId );
        __retiferror(lRet); __closekey(hKey); __closekey(hKey2);
    lRet = regCreate(  base_reg, appid, hAPPIDKey );
        __retiferror(lRet);
    lRet = regSet( hAPPIDKey, L"DllSurrogate", L"" );
        __retiferror(lRet); __closekey(hAPPIDKey);   

    __closekey(hIIDKey);
    __closekey(hCLSIDKey);


    return S_OK;

reg_error:
    __closekey(hInProcSvrKey);
    __closekey(hKey);
    __closekey(hKey2);
    __closekey(hIIDKey);
    __closekey(hAPPIDKey);
    __closekey(hCLSIDKey); 

    return HRESULT_FROM_WIN32( lRet );
#undef __retiferror
#undef __closekey

}

STDAPI DllRegisterServer() {
    return DllRegisterServerCust(false, 0);
}

STDAPI DllUnregisterServerCust(bool all_usr) {
    using namespace o3;
    HKEY base_reg = all_usr ? HKEY_LOCAL_MACHINE : HKEY_CURRENT_USER ;
    
    WStr classes = L"software\\Classes\\";
    WStr progId, progVIId, appGUID;
    WStr clsid, appid, iid;    

    appGUID.appendf(L"%s%s%s", L"{", stemGUID, L"}");
    progId.appendf(L"%s%s%s%s%s", classes.ptr(), baseAppName, L"-", stemGUID, L".1");
    progVIId.appendf(L"%s%s%s%s", classes.ptr(), baseAppName, L"-", stemGUID);
    clsid.appendf(L"%s%s%s", classes.ptr(), L"CLSID\\", appGUID.ptr());
    iid.appendf(L"%s%s%s", classes.ptr(), L"Interface\\",  
        L"{DDBBE8D1-8EE4-4037-816C-42D105D69FF4}");   
    appid.appendf(L"%s%s%s", classes.ptr(), L"APPID\\", appGUID.ptr());     

    LONG lResult = recursiveDeleteKey(base_reg,clsid);
    lResult = recursiveDeleteKey(base_reg, iid);
    lResult = recursiveDeleteKey(base_reg, progId);
    lResult = recursiveDeleteKey(base_reg, progVIId);
    lResult = recursiveDeleteKey(base_reg, appid);
    return S_OK;
}

STDAPI DllUnregisterServer() {
    return DllUnregisterServerCust(false);
}

BOOL WINAPI DllMain(HINSTANCE instance, DWORD fdwReason, LPVOID lpvReserved) {
    using namespace o3;
    switch (fdwReason) {
        case DLL_PROCESS_ATTACH:
        {   
			WSADATA wsd;
			WSAStartup(MAKEWORD(2,2), &wsd);
			if (!g_sys) 
            {
                g_sys = //::new( HeapAlloc(GetProcessHeap(), 0, sizeof(cSys)) ) cSys;
                        new cSys();
                
            }
            g_win32_hinst = instance;     
        } break;

        case DLL_PROCESS_DETACH:
        {   
			WSACleanup();
            if (g_win32_outstanding_objects==0 && g_outerComponents==0 && g_sys) 
            {
                g_sys->release();
                g_sys = 0;
            }
        } break;
    }
   
    return(1);
}

#endif

