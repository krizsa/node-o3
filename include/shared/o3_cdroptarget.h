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

namespace o3{

    struct CDropTarget : 
        IDropTarget
    {
        CDropTarget() 
        {
        }

        virtual ~CDropTarget()
        {        
        }

        mscom_begin(CDropTarget)
			mscom_add_interface(IDropTarget)
		mscom_end();

        //IDropTarget
        HRESULT STDMETHODCALLTYPE DragEnter( IDataObject * pDataObject, DWORD grfKeyState, 
            POINTL pt, DWORD * pdwEffect )
        {
            *pdwEffect = DROPEFFECT_COPY;
            return NOERROR;
        }

        HRESULT STDMETHODCALLTYPE DragOver( DWORD grfKeyState, POINTL pt, DWORD * pdwEffect )
        {
            *pdwEffect = DROPEFFECT_COPY;
            return S_OK;
        }

        HRESULT STDMETHODCALLTYPE DragLeave(void)
        { 
            return E_NOTIMPL; 
        }
        
        HRESULT STDMETHODCALLTYPE Drop( IDataObject * pDataObject, DWORD grfKeyState,
            POINTL pt, DWORD * pdwEffect )
        {
            WStr     file_names;

            FORMATETC fmtetc =
                { CF_HDROP, NULL, DVASPECT_CONTENT, -1, TYMED_HGLOBAL };
            STGMEDIUM stgmed;            
                                    
            if (S_OK == pDataObject->GetData(&fmtetc, &stgmed)) {
                HDROP hdrop = (HDROP)GlobalLock(stgmed.hGlobal);

                if (NULL != hdrop)
                {
                    UINT nFiles = DragQueryFile(hdrop, (UINT)-1, NULL, 0);

                    for(UINT nNames = 0; nNames < nFiles; nNames++)
                    {
                        WStr next_file;
                        next_file.reserve(_MAX_PATH);
                        DragQueryFile
                            (hdrop, nNames, next_file, _MAX_PATH);
                        
                        file_names.append(next_file);
                        file_names.append(L"\n");
                    }
                    GlobalUnlock(hdrop);

                }
                ReleaseStgMedium(&stgmed);

                MessageBoxW(
                    NULL,
                    file_names,
                    L"Files were dropped!",
                    MB_ICONWARNING | MB_OK 
                );
                return (S_OK);
            }
  
            return (S_FALSE);
        }
    };

};
