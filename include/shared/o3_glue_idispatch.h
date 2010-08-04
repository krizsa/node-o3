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

#include "o3_glue.h"

#pragma warning( disable : 4100)
namespace o3 {

	// {3C48E308-C172-4271-894A-F68F394342E3}
	DEFINE_GUID(IID_IDispBridge, 
	0x3c48e308, 0xc172, 0x4271, 0x89, 0x4a, 0xf6, 0x8f, 0x39, 0x43, 0x42, 0xe3);

	// {085DE6F5-A868-4f14-ABFD-BC197AC28269}
	DEFINE_GUID(IID_IDispArray, 
	0x85de6f5, 0xa868, 0x4f14, 0xab, 0xfd, 0xbc, 0x19, 0x7a, 0xc2, 0x82, 0x69);

    struct iScrBridge : iUnk{		
		virtual IDispatchEx* getBridge() = 0;
	};
    o3_iid(iScrBridge, 0x25701b80, 0x8e09, 0x48c3, 0x8c, 0x95, 0x59, 0x35, 0x1e, 0x53, 0xf1, 0xb9);
	
	bool Var_to_VARIANT(const Var &in, VARIANT &out, iCtx* root);
	bool VARIANT_to_Var(const VARIANT &in, Var &out, iCtx* root);

    int incWrapperCount();
    int decWrapperCount();
    struct cScrBridge : cScr, iScrBridge, ComTrack{
        cScrBridge(iCtx* ctx, IDispatchEx* bridge) 
            : ComTrack(siCtx1(ctx)->track()), m_ctx(ctx) , m_disp_bridge(bridge)
		{
            incWrapperCount();
		}
		
		virtual ~cScrBridge() {
            m_disp_bridge = 0;
            decWrapperCount();
        }

		o3_begin_class(cScr)
			o3_add_iface(iScrBridge)
		o3_end_class();

		tMSComPtr<IDispatchEx>  m_disp_bridge;
        siWeak                  m_ctx;

		IDispatchEx* getBridge(){
			return m_disp_bridge;
		};

		siEx invoke(iCtx* ctx, Access access, int index, int va_argc,
				const Var* va_argv, Var* va_rval)
        {
            if (!m_disp_bridge) {
                return o3_new(cEx)(ex_cleanup);
            }

			DISPPARAMS p;
			p.cArgs		 = va_argc;
			p.cNamedArgs = 0;

            p.rgvarg = ::new(g_sys->alloc( sizeof(VARIANTARG) * va_argc )) VARIANTARG[va_argc];
			p.rgdispidNamedArgs = NULL;
			for(uint32_t i = 0;i<p.cArgs;i++)
			{
				VARIANT &v = p.rgvarg[p.cArgs-i-1];
                if ( ! Var_to_VARIANT(va_argv[i],v,siCtx(m_ctx)) ){
                	return o3_new(cEx)(ex_invalid_value);
                }
			}

			VARIANT result;
			result.vt = VT_NULL;
			
			WORD flag;
			switch (access){
				case iScr::ACCESS_CALL: 
					flag = DISPATCH_METHOD; break;
				
				case iScr::ACCESS_GET:  
					flag = DISPATCH_PROPERTYGET; break;
				
				case iScr::ACCESS_SET:  
					flag = DISPATCH_PROPERTYPUT; break;
				
				default : 
					flag = DISPATCH_METHOD; break;
			}		

			if ( S_OK != m_disp_bridge->InvokeEx((DISPID)index, 0, flag, &p, &result, 0, 0) ){
				//!TODO: use an exception struct for the InvokeEx and fetch the error message here
                return o3_new(cEx)("Invoke failed on IDispatchEx.");
			}
			//convert back the result if there is any
			if (va_rval && !VARIANT_to_Var(result,*va_rval, siCtx(m_ctx)) ){
				return o3_new(cEx)(ex_invalid_ret);
			}			
			for (int i=0; i<va_argc; i++){
                VariantClear(&p.rgvarg[i]);
			}
        	VariantClear(&result);
            g_sys->free(p.rgvarg);
            return 0; // Added this to get rid of C4714
		}

        virtual bool invokeExt(iUnk *target, Access type, int id, int o3_argc, const Var* o3_argv, 
                    Var* o3_rval, siEx *o3_ex,iAlloc* alloc){              
                return false;                
        }

		virtual int resolve(iCtx* ctx, const char* name, bool create)
		{
            if (!m_disp_bridge)
                return -1;
			
            // Microsoft apparently use the disp name __self__ and it has id 1
            // but calling InvokeEx with id 1 fails, with id 0 it works fine...
            if (strEquals(name, "__self__"))
                return 0;

			WStr wname(name);			
            DISPID id(0);
			if (wname.size() != 0){
				BSTR bname = SysAllocStringLen(NULL, (UINT)wname.size());
				memCopy((void*)bname,(void*)wname.ptr(),sizeof(wchar_t) * wname.size());
						
                m_disp_bridge->GetDispID(bname,fdexNameEnsure,&id);
				SysFreeString(bname);
			}
			return (int) id;
		}

        int o3::iScr::enumerate(o3::iCtx *,int) {
            //!TODO: implement
            return -1;
        }   

        Trait* select() {
            return 0;
        }

        void tear(){
            m_disp_bridge = 0;
        }

        Str name(iCtx* ctx, int index)
        {
            return Str(); // TODO: Implement
        }
	};

	struct IDispBridge : public IDispatchEx{
		virtual siScr getBridge() = 0;
		
	};
	
	struct CDispBridge : public IDispBridge, ComTrack{
        CDispBridge(iCtx1 *ctx, iScr *bridge) : ComTrack(ctx->track()) 
		{
            // db_mtrace();
            m_bridge = bridge;
            m_ctx = ctx;            
		}
		
        siWeak      m_ctx;
        siScr		m_bridge;

		virtual ~CDispBridge() {
            // db_mtrace();
            //incWrapperCount();
		}

		siScr getBridge(){
            // db_mtrace();
            //decWrapperCount();
			return m_bridge;
		}

		//IUnknown:
		mscom_begin_debug(CDispBridge)
			mscom_add_interface(IDispatch);
			mscom_add_interface(IDispatchEx)
			//mscom_add_interface(IObjectSafety)
			mscom_add_interface(IDispBridge)
		mscom_end();

        ULONG STDMETHODCALLTYPE AddRef() {
            // db_mtrace();
            int32_t ret = atomicInc(_m_com.ref_count);
            if (ret == 1)
                incWrapperCount();
            return (ULONG)ret;
        } 
        ULONG STDMETHODCALLTYPE Release() {              
            // db_mtrace();
            int ret = atomicDec(_m_com.ref_count);
            if( ret == 0){ 
                decWrapperCount();
                this->~CDispBridge(); 
                g_sys->free(this); 
            } 
            return (ULONG)ret;
        } 		

		HRESULT  STDMETHODCALLTYPE GetInterfaceSafetyOptions(	REFIID riid,   
																DWORD *pdwSupportedOptions,   
																DWORD *pdwEnabledOptions	)
		{
            // db_mtrace();
			*pdwSupportedOptions = INTERFACESAFE_FOR_UNTRUSTED_DATA|INTERFACESAFE_FOR_UNTRUSTED_CALLER;
			*pdwEnabledOptions	 = INTERFACESAFE_FOR_UNTRUSTED_DATA|INTERFACESAFE_FOR_UNTRUSTED_CALLER;
			return S_OK;
		}
		
		//IDispatch:
		HRESULT STDMETHODCALLTYPE SetInterfaceSafetyOptions( REFIID riid, 
			DWORD dwOptionSetMask, DWORD dwEnabledOptions	) { // db_mtrace();
                return E_NOTIMPL; }
		
		HRESULT STDMETHODCALLTYPE GetTypeInfoCount(UINT* pctinfo)
		{
            // db_mtrace();
			*pctinfo = 0;
			return S_OK;
		}

		HRESULT STDMETHODCALLTYPE GetTypeInfo(UINT itinfo, LCID lcid, ITypeInfo** pptinfo) { // db_mtrace();
            return E_NOTIMPL; }		
		HRESULT STDMETHODCALLTYPE Invoke(DISPID dispidMember, REFIID riid, LCID lcid, WORD wFlags,
			DISPPARAMS* pdispparams, VARIANT* pvarResult, EXCEPINFO* pexcepinfo, UINT* puArgErr){ // db_mtrace();
                return E_NOTIMPL; }
		HRESULT STDMETHODCALLTYPE GetIDsOfNames(REFIID riid, LPOLESTR* rgszNames, UINT cNames,
			LCID lcid, DISPID* rgdispid){ // db_mtrace();
                return E_NOTIMPL; }				

		//IDispatchEx
		HRESULT STDMETHODCALLTYPE DeleteMemberByDispID(DISPID id)
		{		
            // db_mtrace();
            siEx ex = m_bridge->invoke(siCtx(m_ctx),iScr::ACCESS_DEL, id, 
                0,0,0); 
            
            return ex ? S_FALSE : S_OK;		
		}

		HRESULT STDMETHODCALLTYPE DeleteMemberByName(BSTR bstrName, DWORD grfdex)
		{
            // db_mtrace();
			DISPID id;
			GetDispID(bstrName, grfdex, &id);
			return DeleteMemberByDispID(id);
		}

		#define __ARRAY_IDX_DISPID 2000
		HRESULT STDMETHODCALLTYPE GetDispID(BSTR bstrName, DWORD grfdex, DISPID *pid)
		{
            if(!m_bridge)
			return S_FALSE;
            // db_mtrace();
			//is it an index?
			wchar_t* it = bstrName;
			if ( chrIsDigit(*it))
			{
				++it;
				while(*it){					
					if ( ! chrIsDigit(*it))
						return DISP_E_BADINDEX;
					++it;
				}
				*pid = strToInt32(bstrName) + __ARRAY_IDX_DISPID;
				return S_OK;
			}

            //TOCHECK:
			Str name((wchar_t*)bstrName);
			*pid = (int) m_bridge->resolve(siCtx(m_ctx), name.ptr(),(fdexNameEnsure & grfdex) != 0 );
            return (*pid > -1) ? S_OK :  DISP_E_UNKNOWNNAME;
		}

		HRESULT STDMETHODCALLTYPE GetMemberName(DISPID id, BSTR *pbstrName)
		{		
            if(!m_bridge)
                return S_FALSE;
            
            if (id >= __ARRAY_IDX_DISPID) {
                //TODO: what about arrays??
                return S_FALSE;
            }

            int scr_id = m_bridge->enumerate(siCtx(m_ctx), id-1);
            ScrInfo info;
            scrInfo(m_bridge, siCtx(m_ctx), scr_id, &info);

            *pbstrName = SysAllocString(WStr(info.name).ptr());
            return S_OK;
		}

		HRESULT STDMETHODCALLTYPE GetMemberProperties(DISPID id, DWORD grfdexFetch, DWORD *pgrfdex){//db_assert(false); 
            return(E_NOTIMPL);}
		HRESULT STDMETHODCALLTYPE GetNameSpaceParent(IUnknown **ppunk){//db_assert(false);
            return(E_NOTIMPL); }

		HRESULT STDMETHODCALLTYPE GetNextDispID(DWORD grfdex, DISPID id, DISPID *pid){  
            // db_mtrace();
            
            if(!m_bridge)
                return S_FALSE;

            if (id >= __ARRAY_IDX_DISPID) {
                //TODO: what about arrays??
                return S_FALSE;
            }

            *pid = (int) m_bridge->enumerate(siCtx(m_ctx), (int) id);

            return (E_NOTIMPL);
        }

		HRESULT STDMETHODCALLTYPE InvokeEx(	DISPID id, LCID lcid, WORD wFlags, 
											DISPPARAMS *pdp, VARIANT *pVarRes, 
											EXCEPINFO *pei, IServiceProvider *pspCaller){
            // db_mtrace();
			if(!m_bridge)
			return S_FALSE;
			int call_id;
            const char* array_acc("__getter__");
			iScr::Access call_type;
			//set the call type
			if ((wFlags & DISPATCH_METHOD)){						
				call_type = iScr::ACCESS_CALL;                
			
			}else if (wFlags & DISPATCH_PROPERTYGET){
				call_type = iScr::ACCESS_GET;
			
			}else if (wFlags &	(DISPATCH_PROPERTYPUT | DISPATCH_PROPERTYPUTREF)){
				call_type = iScr::ACCESS_SET;
                array_acc = "__setter__";
			}else{
				return DISP_E_EXCEPTION;
			}
			//getting the arguments
			UINT nargs = pdp ? pdp->cArgs : 0;	
			tVec<Var> args;
			if (id >= __ARRAY_IDX_DISPID)
			{
				//array (call by _item_acc) 
				if (call_type == iScr::ACCESS_SET)
				{
                    args.append(Var(Var::TYPE_VOID,g_sys),2);
					args[0].set((id - __ARRAY_IDX_DISPID));
					VARIANT_to_Var( pdp->rgvarg[nargs-1], args[1], siCtx(m_ctx));				        
                }
				else if (call_type == iScr::ACCESS_GET)
				{
					args.append(Var((int) id - __ARRAY_IDX_DISPID, g_sys));
				}
				else return DISP_E_BADCALLEE;	
                call_id = m_bridge->resolve(siCtx(m_ctx),array_acc, false);
                call_type = iScr::ACCESS_CALL;
			}
			else
			{
				//non array (call by invoke)
				args.append(Var(Var::TYPE_VOID, g_sys),nargs);
				for(uint32_t i = 0;i<nargs;i++)
				{
					VARIANT& v = pdp->rgvarg[nargs-i-1];				
					if (!VARIANT_to_Var(v, args[i],siCtx(m_ctx)))													
						return DISP_E_BADVARTYPE;
				}
                call_id = (int)id;
			}

			siScrFun fun = m_bridge;
			size_t call_argc = fun ? args.size()-1 : args.size();
			Var ret(g_sys);						
            siEx ex = m_bridge->invoke(siCtx(m_ctx), call_type, call_id,
                                (int) call_argc, args.ptr(), &ret);
            if (ex) {
                if (!ex) return DISP_E_BADCALLEE;
                WStr msg(L"O3 callback for function returned an error : ");
                msg.append(WStr(ex->message()));
                pei->bstrSource = SysAllocString(L"O3 exception");
                pei->bstrDescription = SysAllocString(msg.ptr());
                pei->wCode = 1001;
                return DISP_E_EXCEPTION;
            }					
			

			// setting the return value
            if (pVarRes)
                if (!Var_to_VARIANT(ret,*pVarRes,siCtx(m_ctx)) )
				    return DISP_E_BADVARTYPE;
			return S_OK;
		}

        void tear(){
            // db_mtrace();
            m_bridge = (iUnk*)0;
        }

	};//cDispBridge

    //Utility functions
	bool VARIANT_to_Var(const VARIANT &in, Var &out, iCtx* ctx){
		switch(in.vt)
		{	
			case VT_I1:     // Char.
			case VT_UI1:    // Unsigned char.
			case VT_UI2:    // 2 byte unsigned int.
			case VT_UI4:    // 4 byte unsigned int. 
			case VT_INT:    // Signed machine int.
			case VT_UINT:   // Unsigned machine int.
			case VT_I2:		
			case VT_I4:     // 4-byte-signed int.
			case VT_HRESULT: 
				out.set(in.intVal);
				break;                               
			
			case VT_R4:      
				out.set((double) in.fltVal);
				break;
			
			case VT_R8:      
				out.set(in.dblVal);
				break;

			case VT_I8:		 
				// out = (int64_t) in.llVal; // INT64 disabled for release			
				out = (int) in.llVal;
                break;
			
			case VT_BSTR:       // Automation string.
			case VT_LPWSTR:     // Wide null-terminated string.
			{
				//int32_t wlen = SysStringLen(in.bstrVal);				

                out = Str(WStr(in.bstrVal));
			}
			break;
			
			case VT_LPSTR:      // Null-terminated string.
				out.set( in.pcVal); 
				break;
			
			case VT_DISPATCH:
			{  
				IDispBridge *bridge = NULL;
				IDispatchEx *dispe = NULL; 				
				if(!in.pdispVal)
                    return false;				
				in.pdispVal->QueryInterface(IID_IDispBridge, (void**)&bridge);
				if (bridge){
					//unwrap the object
					siScr sc( bridge->getBridge() );
                    out.set(sc.ptr());
					bridge->Release();
					break;
				}
				in.pdispVal->QueryInterface(IID_IDispatchEx, (void**)&dispe);					
				if (dispe){
					//if it is not a IDispBirdge, then we have to wrap it
                    cScrBridge* pscBr = o3_new(cScrBridge)(ctx,dispe);                    
                    out = pscBr;
                    dispe->Release();
					break;
				}									
			}
										
			case VT_BOOL:        // Boolean; True=-1, False=0.
				out.set( in.boolVal!=0 ); 
				break;
			
			case VT_ERROR:
			case VT_NULL:

			case VT_EMPTY:
				if ( out.type() != Var::TYPE_VOID){
					out = 0;
				}				
				break;
			
            case VT_UNKNOWN:       // IUnknown FAR*.
			case VT_VOID:					// C-style void.
			case VT_PTR:					// Pointer type.
			case VT_VARIANT:				// VARIANT FAR*.
			case VT_DECIMAL:				// 16 byte fixed point.
			case VT_RECORD:					// User defined type
			case VT_CY:						// Currency.
			case VT_DATE:					// Date.
			case VT_SAFEARRAY:              // Use VT_ARRAY in VARIANT.
			case VT_CARRAY:					// C-style array.
			case VT_USERDEFINED:			// User-defined type.
			case VT_FILETIME:				// FILETIME
			case VT_BLOB:					// Length prefixed bytes
			case VT_STREAM:					// Name of the stream follows
			case VT_STORAGE:				// Name of the storage follows
			case VT_STREAMED_OBJECT:		// Stream contains an object
			case VT_STORED_OBJECT:          // Storage contains an object
			case VT_BLOB_OBJECT:			// Blob contains an object
			case VT_CF:						// Clipboard format
			case VT_CLSID:					// A Class ID
			case VT_VECTOR:					// simple counted array
			case VT_BYREF:
			default:	 return false;
		}
		return true;
	}


	bool Var_to_VARIANT(const Var &in, VARIANT &out, iCtx* ctx){
		VariantInit(&out);
		switch(in.type())
		{
            case Var::TYPE_NULL:
			case Var::TYPE_VOID:
				out.vt = VT_EMPTY;
				break;

			case Var::TYPE_INT32:	
				out.vt = VT_INT;
				out.lVal = in.toInt32();
				break;

            case Var::TYPE_INT64:
                out.vt = VT_I8;
                out.llVal = in.toInt64(); 
            

			case Var::TYPE_DOUBLE:
				out.vt = VT_R8;
				out.dblVal = in.toDouble();
				break;

			case Var::TYPE_BOOL:
				out.vt = VT_BOOL;
				out.boolVal = in.toBool();
				break;

            case Var::TYPE_STR:	{
                WStr ret(in.toStr());
				out.vt = VT_BSTR;
				out.bstrVal = SysAllocString(ret.ptr()); 
				break;
                                }
			case Var::TYPE_WSTR:

				out.vt = VT_BSTR;
				out.bstrVal = SysAllocString(in.toWStr().ptr()); 
				break;
	
			case Var::TYPE_SCR:
			{
				bool ret = false;
				// lets see if this is a wrapped IDispatch
				siScrBridge bridge = in.toScr();				
				//in.toIScr()->queryInterface( __get_guid(IID_iScrBridge) , (void**)&bridge );
                if(bridge)
				{
					//unwrap the object
					IDispatch* sc = bridge->getBridge();
					//sc->AddRef();
					out.vt=VT_DISPATCH;
					out.pdispVal = sc;
					ret = true;
                }else{				
					siScr iscr = in.toScr();													
					CDispBridge* ib = o3_new(CDispBridge)(siCtx1(ctx),iscr);
					out.vt = VT_DISPATCH;
					out.pdispVal = ib;
					ib->AddRef();
					ret = true;					
				}
			}
			break;

			default:	 
				return false;
		}
		return true;
	}



}//namespace o3
#pragma warning(default : 4100)