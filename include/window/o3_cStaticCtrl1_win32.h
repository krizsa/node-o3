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
#ifndef O3_C_STATIC_CTRL1_WIN32_H
#define O3_C_STATIC_CTRL1_WIN32_H

namespace o3 {

struct cStaticCtrl1 : cWindow1
{
    enum StaticType {
        TYPE_TEXTBOX,
        TYPE_IMAGEBOX, 
        TYPE_BLANK, 
        TYPE_SEPARATOR
    };

    cStaticCtrl1(){}
    virtual ~cStaticCtrl1(){}

    o3_begin_class(cWindow1)
        o3_add_iface(iWindow)
        o3_add_iface(iWindowProc)
    o3_end_class()

    o3_glue_gen()

    StaticType      m_type;
    WNDPROC         m_def_proc;
    HBITMAP         m_bitmap;    

    static o3_ext("cWindow1") o3_fun siWindow createTextbox(o3_tgt iScr* target, const char* text, 
        int x, int y, int w, int h, int font_size = 16, int font_style = 0, int text_color = -1)
    {
        return createStatic(TYPE_TEXTBOX,target,text,x,y,w,h,font_size,font_style,text_color);
    }

    static o3_ext("cWindow1") o3_fun siWindow createBlank(o3_tgt iScr* target,  
        int x, int y, int w, int h, int color=0)
    {
        return createStatic(TYPE_BLANK,target,0,x,y,w,h,16,0,color);
    }

    static o3_ext("cWindow1") o3_fun siWindow createSeparator(o3_tgt iScr* target, 
        int x, int y, int w)
    {
        return createStatic(TYPE_SEPARATOR,target,0,x,y,w,3);
    }

    static o3_ext("cWindow1") o3_fun siWindow createImgbox(o3_tgt iScr* target, const char* img_name, 
        int x, int y, int w, int h)
    {
        return createStatic(TYPE_IMAGEBOX,target,img_name,x,y,w,h);
    }

    static siWindow createStatic(StaticType type, iScr* target,
        const char* text_data, int x, int y, int w, int h, int font_size = 16, int font_style = 0, int color = -1)
    {
        // create component
        cStaticCtrl1* ret = o3_new(cStaticCtrl1)();
        ret->m_type = type;

        siWindow parent = target;
        HWND parent_hwnd = parent ? (HWND) parent->handle() : NULL;

        DWORD extra_flags(0),flags = WS_CHILD | WS_VISIBLE;        
        WStr text;        
        switch (type)
        {
            case TYPE_IMAGEBOX:
                extra_flags = SS_BITMAP;
                ret->loadBitmapFromRsc(text_data);
                break;
            case TYPE_BLANK:
                ret->m_color = color;           
                break;
            case TYPE_SEPARATOR:
                ret->m_color = LTGRAY_BRUSH;
                extra_flags = SS_ETCHEDHORZ;
                break;
            case TYPE_TEXTBOX:
                text = text_data;
				ret->m_text_color = color;
            default:
                ret->m_color = NULL_BRUSH;
                break;        
        }

        flags |= extra_flags;

        ret->m_hwnd = CreateWindowW(L"STATIC", L"", flags, x, y, w, h,
                parent_hwnd, 0, GetModuleHandle(0), 0);

        if (type == TYPE_TEXTBOX) {
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
            SetWindowTextW( ret->m_hwnd, text);
        }

        if (ret->m_bitmap) {
            SendMessage(ret->m_hwnd, STM_SETIMAGE,  
                (WPARAM)IMAGE_BITMAP, (LPARAM)ret->m_bitmap);                
        }
           
        SetWindowLongPtr( ret->m_hwnd, GWL_USERDATA, (LONG_PTR)(iWindowProc*)ret );
        ret->m_def_proc = (WNDPROC) SetWindowLongPtr (ret->m_hwnd, GWL_WNDPROC,
            (LONG_PTR)(WNDPROC)_WndProc);

        return ret;
    }

    virtual LRESULT CALLBACK wndProc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam)
    {
        return m_def_proc(hwnd,message,wparam,lparam);
    }

    bool loadBitmapFromRsc(const char* name)
    {
        Buf img = ((cSys*) g_sys)->resource(name);
        if (img.empty())
            return false;

        WStr file_name = tempFile(img);
        if (file_name.empty())
            return false;

        m_bitmap = (HBITMAP) LoadImageW ( NULL, WStr(Str(file_name)), IMAGE_BITMAP, 0, 0,
               LR_CREATEDIBSECTION | LR_DEFAULTSIZE | LR_LOADFROMFILE );
        
        return (m_bitmap != NULL);
    }
};

}

#endif // O3_C_STATIC_CTRL1_WIN32_H
