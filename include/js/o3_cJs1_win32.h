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
#ifndef O3_C_JS1_WIN32_H
#define O3_C_JS1_WIN32_H

#include "shared/o3_tools_win32.h"
#include "shared/o3_glue_idispatch.h"

namespace o3{
    volatile int32_t g_outerComponents = 0;

    int incWrapperCount() {
        return atomicInc(g_outerComponents);
    }

    int decWrapperCount() {
        return atomicDec(g_outerComponents);
    } 

}

#include <activscp.h>
#include "../external/include/JScript/activscp_plus.h"
#include "core/o3_cO3.h"

namespace o3{

	// CLSID for ProcessDebugManager
	// {78A51822-51F4-11D0-8F20-00805F2CD064}
	static const GUID CLSID_ProcessDebugManager =
	{ 0x78a51822, 0x51f4, 0x11d0, { 0x8f, 0x20, 0x0, 0x80, 0x5f, 0x2c, 0xd0, 0x64 } };

	static const GUID IID_IActiveScriptSiteDebug32 =
	{ 0x51973c11, 0xcb0c, 0x11d0, {0xb5, 0xc9, 0x00 ,0xa0 ,0x24 ,0x4a ,0x0e ,0x7a} };

	static const GUID IID_IDebugApplication32 =
	{0x51973c32, 0xcb0c, 0x11d0, {0xb5, 0xc9, 0x00, 0xa0, 0x24, 0x4a, 0x0e, 0x7a}};

	typedef IActiveScriptSiteDebug IActiveScriptSiteDebug32;	
    struct cJs1 : cScr, iCtx, iCtx1 {
        o3_cls(cJs1);

		struct CAScript :			
			public IActiveScriptSite,
			public IActiveScriptSiteDebug {

            CAScript() : m_debug(false){
			}

			virtual ~CAScript() {
                deinit();
			}

			mscom_begin(CAScript)
				mscom_add_interface(IActiveScriptSite)
				mscom_add_interface(IActiveScriptSiteDebug32)
			mscom_end();


            scJs1							m_pthis;
			tMSComPtr<IActiveScript>		m_activeScript;
			tMSComPtr<IActiveScriptParse>	m_activeScriptParse;
			tMSComPtr<IUnknown>				m_bridge;
            tMSComPtr<IUnknown>				m_inc_bridge;
			WStr							m_root_name;

			//debugger
			tMSComPtr<IProcessDebugManager>	m_pdm;
			tMSComPtr<IDebugApplication>	m_debug_app; 
			tMSComPtr<IDebugDocumentHelper>	m_debug_doc_helper; 
			DWORD							m_app_cookie;
            bool                            m_debug;

			bool init(cJs1 *ipthis, bool debug) {
                m_debug = debug;
				m_pthis = ipthis;
				HRESULT hr = S_OK;
				tMSComPtr<IActiveScriptSite>  activeScriptSite = 0;

				GUID CLSID_JScript  = 
                    {0xf414c260, 0x6ac0, 0x11cf, {0xb6, 0xd1, 0x00, 0xaa, 0x00, 0xbb, 0xbb, 0x58}};
                
                if (m_debug) {
				    //create debugger
				    hr = CoCreateInstance(CLSID_ProcessDebugManager,
					      NULL,
					      CLSCTX_INPROC_SERVER,
					      IID_IProcessDebugManager,
					      (void**)&m_pdm);
				    if (FAILED(hr))
					    return false;
                }
				hr = CoCreateInstance(CLSID_JScript,
					NULL,
					CLSCTX_INPROC_SERVER,
					IID_IActiveScript,
					(void **)&m_activeScript);
				if (FAILED(hr))
					return false;

				hr = m_activeScript->QueryInterface( IID_IActiveScriptParse, 
                                                     (void **)&m_activeScriptParse);
				if (FAILED(hr))
					return false;
            
                if (m_debug) {
          
				    hr = m_pdm->CreateApplication(&m_debug_app);
				    if (FAILED(hr))
					    return false;
				    //m_debug_app->AddRef();

				    hr = m_debug_app->SetName(L"O3 Scripting Application");
				    if (FAILED(hr))
					    return false;				

				    hr = m_pdm->AddApplication(m_debug_app, &m_app_cookie);
				    if (FAILED(hr))
					    return false;

				    hr = m_pdm->CreateDebugDocumentHelper(NULL, &m_debug_doc_helper);
				    if (FAILED(hr))
					    return false;

				    hr = m_debug_doc_helper->Init(m_debug_app, L"Test Js", L"Long Name of Test Js", 
                                                  TEXT_DOC_ATTR_READONLY);
				    if (FAILED(hr))
					    return false;

				    hr = m_debug_doc_helper->Attach(NULL);
				    if (FAILED(hr))
					    return false;
                }

				hr = this->QueryInterface(IID_IActiveScriptSite, (void **)&activeScriptSite);
				if (FAILED(hr))
					return false;			

				hr = m_activeScript->SetScriptSite(activeScriptSite);
				if (FAILED(hr))
					return false;
				                
				return true;
			}

            void deinit(){
                m_activeScript->Close();
            
                if (m_debug) {
                    //releasing the debugger
                    m_debug_doc_helper->Detach();
                    m_debug_doc_helper.Release();
				    m_pdm->RemoveApplication(m_app_cookie);
                }
            }

			public:
			// IActiveScriptSite
			HRESULT STDMETHODCALLTYPE GetLCID(LCID *){ return E_NOTIMPL; }
			HRESULT STDMETHODCALLTYPE GetDocVersionString(BSTR *) {
                return E_NOTIMPL; }
			HRESULT STDMETHODCALLTYPE OnScriptTerminate(const VARIANT *,
												const EXCEPINFO *){ return E_NOTIMPL; }
			HRESULT STDMETHODCALLTYPE OnStateChange(SCRIPTSTATE ){ return E_NOTIMPL;}
			HRESULT STDMETHODCALLTYPE OnEnterScript(void){ return E_NOTIMPL; }
			HRESULT STDMETHODCALLTYPE OnLeaveScript(void){ return E_NOTIMPL; }

			HRESULT STDMETHODCALLTYPE OnScriptError(IActiveScriptError *error){	
                EXCEPINFO excepinfo;
                error->GetExceptionInfo(&excepinfo);
                
                DWORD dwSourceContext;  
                ULONG ulLineNumber;     
                LONG ichCharPosition;

                error->GetSourcePosition(&dwSourceContext,&ulLineNumber,&ichCharPosition);
                if (!excepinfo.bstrDescription)
                    return E_FAIL;
                
                m_pthis->m_error.appendf("%s%s%s%d%s%d\n", "ScriptError: ", 
                    Str(excepinfo.bstrDescription).ptr(), 
                    ",   line: ", ulLineNumber, ",   char: ", ichCharPosition);
                
                fprintf(stderr,m_pthis->m_error.ptr());                
                //return E_FAIL;
                return S_OK;
			}

			// IActiveScriptHost
			// This is called for each named item. Look into the map and return
			HRESULT STDMETHODCALLTYPE GetItemInfo(LPCOLESTR name,
															 DWORD ,
															 IUnknown **item,
															 ITypeInfo **)
            {							
                if (strEquals(m_root_name.ptr(), name)){
                    *item = m_bridge;
                    m_bridge->AddRef();
					return S_OK;
                }

                return TYPE_E_ELEMENTNOTFOUND;			
			}

			// Inject the root object
			HRESULT STDMETHODCALLTYPE Inject(const WCHAR *name, IUnknown *unkn) {
				// db_assert(name != NULL);

				if (name == NULL)
					return E_POINTER;
				if ( ! m_root_name.empty()) 
                    return S_OK;					
				m_activeScript->AddNamedItem(name, SCRIPTITEM_GLOBALMEMBERS | SCRIPTITEM_ISVISIBLE );
				m_root_name = WStr(name);
				m_bridge = unkn;
				return S_OK;
			}

			// Evaluation routine.
			HRESULT STDMETHODCALLTYPE Eval(const WCHAR *source, VARIANT *result) {
				// db_assert(source != NULL);
				if (source == NULL)
					return E_POINTER;

				HRESULT hr;
                DWORD dw = 0;
                
                if (m_debug) {
                    //setting up the installer for the script
                hr = m_debug_doc_helper->AddUnicodeText(source);
				if (!SUCCEEDED(hr))
				    return S_FALSE;  
				
				hr = m_debug_doc_helper->DefineScriptBlock(0, (ULONG)strLen(source), m_activeScript, FALSE, &dw);
				if (!SUCCEEDED(hr))
                    return S_FALSE;  
                }
                /*SCRIPTTEXT_ISVISIBLE*/ /*SCRIPTTEXT_HOSTMANAGESSOURCE*/ /* SCRIPTTEXT_ISEXPRESSION */
				HRESULT ret = m_activeScriptParse->ParseScriptText(source, NULL, NULL, NULL, dw, 1,
                      SCRIPTTEXT_ISEXPRESSION , result, NULL);
				return ret;
			}
 
			HRESULT STDMETHODCALLTYPE GetDocumentContextFromPosition( DWORD dwSourceContext,
				  ULONG uCharacterOffset, ULONG uNumChars, IDebugDocumentContext **ppsc)
			{
			   ULONG ulStartPos = 0;
			   HRESULT hr;

			   if (m_debug_doc_helper)
			   {
				  hr = m_debug_doc_helper->GetScriptBlockInfo(dwSourceContext, NULL, &ulStartPos, NULL);
				  hr = m_debug_doc_helper->CreateDebugDocumentContext(ulStartPos + uCharacterOffset, uNumChars, ppsc);
			   }
			   else
				  hr = E_NOTIMPL;

				return hr;
			}

			HRESULT STDMETHODCALLTYPE GetApplication(IDebugApplication **ppda)
			{
			   if (!ppda)
				  return E_INVALIDARG;
			  			   
			   ULONG ul;
               if (m_debug_app){
			   	  ul = m_debug_app->AddRef();
			   
			      *ppda = m_debug_app;
               }else
                   return E_NOTIMPL;

				return S_OK;
			}

			HRESULT STDMETHODCALLTYPE GetRootApplicationNode(IDebugApplicationNode **ppdanRoot)
			{
			   if (!ppdanRoot)
				  return E_INVALIDARG;
			   if (m_debug_doc_helper)
				  return m_debug_doc_helper->GetDebugApplicationNode(ppdanRoot);
			   
               return E_NOTIMPL;
			}
			HRESULT STDMETHODCALLTYPE OnScriptErrorDebug( IActiveScriptErrorDebug *, 				  
				  BOOL*pfEnterDebugger, BOOL *pfCallOnScriptErrorWhenContinuing)
			{
			   if (pfEnterDebugger)
			   	  *pfEnterDebugger = TRUE;		

			   if (pfCallOnScriptErrorWhenContinuing)
			   	  *pfCallOnScriptErrorWhenContinuing = TRUE;
			   return S_OK;
			}

		};	

        cJs1(iMgr* mgr = o3_new(cMgr)(), int argc=0, char** argv=0, char** envp=0, bool debug = false) 
            : m_mgr(mgr), m_track(0), m_loop(g_sys->createMessageLoop())
        {
            m_root = o3_new(cO3)(this, argc,argv,envp);
            initEngine(debug);
        }

        void initEngine(bool debug)
        {
            //bridge
            CDispBridge* bridge = o3_new( CDispBridge )(this,m_root);
            m_ascript = o3_new(CAScript)();
            if(m_ascript->init(this, debug)){                                     
                //injecting our root object into the global object            
                IUnknown* unk;
                bridge->QueryInterface(IID_IUnknown, (void**) &unk);
                m_ascript->Inject(L"o3", unk);                              
                unk->Release();
            }        
        }

		virtual ~cJs1(){
            if (m_ascript)
                m_ascript.Release();
        }		

		o3_begin_class(cScr)
            o3_add_iface(iAlloc)
            o3_add_iface(iCtx)
            o3_add_iface(iCtx1)
        o3_end_class();

        o3_glue_gen()

        static o3_ext("cO3") o3_get siScr js(iCtx* ctx)
        {
            // TODO: this should be created only once per ctx
            // siJs js = o3_new(cJs1)(ctx->mgr());            
            return siScr(ctx);
        }

        virtual o3_fun Var include(const char* path, siEx* ex = 0) 
        {
            Str str = eval(Str("o3.cwd.get(\"") + path + "\").data", ex).toStr();
            Var res = eval(str, ex);

            if (*ex) {
                o3_set_ex((*ex)->message() + " in file " + path);
                return Var();
            }
            return res;
        }        

		virtual o3_fun Var eval(const char* src, siEx* ex)
        {
            ex;
            Var vret;
            VARIANT result;
			WStr script;
            result.vt = VT_EMPTY;
            if (*src == '#')
            while (*src)
                if (*src++ == '\n')
                    break;
            script = WStr(src);

            HRESULT res = m_ascript->Eval(script, &result);
            
            if ( res < 0 )
                return m_error;
            else 
                VARIANT_to_Var(result, vret, siCtx(this));            

            return vret;                        
		}

        void* alloc(size_t size)
        {
            return g_sys->alloc(size);
        }

        void free(void* ptr)
        {
            return g_sys->free(ptr);
        }

        siMgr mgr()
        {
            return m_mgr;
        }

        virtual ComTrack** track() {
           return &m_track;
        }

        virtual siMessageLoop loop()
        {
            return m_loop;
        }

        virtual Var value(const char* key) 
        {
            return m_values[key];
        }

        virtual Var setValue(const char* key, const Var& val)
        {
            return m_values[key] = val;
        }

        virtual void tear() {
            for (ComTrack *i = m_track, *j = 0; i; i = j) {
                j = i->m_next;
                i->m_phead = 0;
                i->tear();
            }
        }

        virtual Str fsRoot()
        {
            return Str();
        }

        virtual void setAppWindow(void*)
        {
            // TODO: this function pair should not be on the context
        }

        virtual void* appWindow()
        {
            return 0;
        }

		bool scriptError() 
		{
			return m_error.size() > 0;
		}
		


        tMSComPtr<CAScript>         m_ascript;     
        HRESULT                     m_hrInit;
        siScr                       m_root;
        Str                         m_error;
        siMgr                       m_mgr;
        ComTrack*                   m_track;
        siMessageLoop               m_loop;
        tMap<Str, Var>              m_values;
	};

}


#endif // O3_C_JS1_WIN32_H
