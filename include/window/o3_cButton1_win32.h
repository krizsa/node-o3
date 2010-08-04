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
#ifndef O3_C_BUTTON1_WIN32_H
#define O3_C_BUTTON1_WIN32_H

namespace o3 {

struct cButton1 : cWindow1
{
    enum ButtonType {
        TYPE_PUSH,
        TYPE_RADIO 
    };

    cButton1(){}
    virtual ~cButton1(){}

    o3_begin_class(cWindow1)
        o3_add_iface(iWindow)
        o3_add_iface(iWindowProc)
    o3_end_class()

    o3_glue_gen()

    ButtonType      m_type;
    WNDPROC         m_def_proc;
    siScr           m_onclick;

    static o3_ext("cWindow1") o3_fun siWindow createButton(o3_tgt iScr* target, const char* text, 
        int x, int y, int w, int h, int font_size = 16, int font_style = 0)
    {              
        return create(TYPE_PUSH,target,text,x,y,w,h,font_size,font_style, 0);
    }

    static o3_ext("cWindow1") o3_fun siWindow createRButton(o3_tgt iScr* target, const char* text, 
        int x, int y, int w, int h, int font_size = 16, int font_style = 0, int bkcolor = 0)
    {
        return create(TYPE_RADIO,target,text,x,y,w,h,font_size,font_style, bkcolor);
    }

    static siWindow create(ButtonType type, iScr* target, const char* text,
        int x, int y, int w, int h, int font_size, int font_style, int bkcolor)
    {
        // create the component
        cButton1* ret = o3_new(cButton1)();
        ret->m_type = type;

        DWORD flags = WS_CHILD | WS_VISIBLE | WS_TABSTOP | WS_SYSMENU | BS_PUSHBUTTON;
        if (type == TYPE_RADIO) {
            flags |= BS_AUTORADIOBUTTON;
            ret->m_color = bkcolor;
        }

        siWindow parent = target;
        HWND parent_hwnd = parent ? (HWND) parent->handle() : NULL;

        // create the window
        ret->m_hwnd = CreateWindowExW(0,L"BUTTON", WStr(Str(text)), flags, x, y, 
            w, h, parent_hwnd, 0, GetModuleHandle(0), (LPVOID)ret);        
     
        // create font
        HFONT dfont = CreateFontW( font_size, 0, 0, 0, 
                                        (font_style & FONT_BOLD) ? FW_BOLD : FW_NORMAL, 
                                        (font_style & FONT_ITALIC), 
                                        (font_style & FONT_UNDERLINE), 
                                        (font_style & FONT_STRIKEOUT),
									    DEFAULT_CHARSET,OUT_DEFAULT_PRECIS,CLIP_DEFAULT_PRECIS,
									ANTIALIASED_QUALITY,FF_DONTCARE,L"Tahoma");
	    // set font
        SendMessage(ret->m_hwnd ,WM_SETFONT,(WPARAM)dfont,(LPARAM)TRUE);

        // overwriting the original button proc function and storing the original for later usage
        SetWindowLongPtr( ret->m_hwnd, GWL_USERDATA, (LONG_PTR)(iWindowProc*)ret );
        ret->m_def_proc = (WNDPROC)SetWindowLongPtr(ret->m_hwnd, 
            GWL_WNDPROC,(LONG_PTR)(WNDPROC)_WndProc);

        return ret;
    }

    virtual LRESULT CALLBACK wndProc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam)
    {
        siCtx ctx(m_ctx);
        switch(message){
            case WM_COMMAND:{
                switch(LOWORD(wparam)){
                    case BN_CLICKED:     
                        // if onlick prop set then and the button was pressed, 
                        // m_onclick needs to get called back
                        if (m_onclick && ctx) {
                            Var ret(g_sys);                     
                            int id = m_onclick->resolve(ctx,"__self__",false);
                            m_onclick->invoke(ctx, ACCESS_CALL, id, 0, 0, &ret);
                        }                            
                    break;
                }
            }break;
            default:
                return m_def_proc(hwnd,message,wparam,lparam);
        }
        return 0;
    }

    o3_set siScr setOnclick(iCtx* ctx, iScr* cb)
    {
        m_ctx = ctx;
        return m_onclick = cb;
    }
};

}

#endif // O3_C_BUTTON1_WIN32_H
