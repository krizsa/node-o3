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
#ifndef O3_C_WINDOW1_APPLE_H
#define O3_C_WINDOW1_APPLE_H

#include <Cocoa/Cocoa.h>

namespace o3 {

struct cWindow1 : cWindow1Base {
    static o3_ext("cO3") o3_fun siScr window(iCtx* ctx)
    {
        static Var window = ctx->value("window");
        
        if (window.type() == Var::TYPE_VOID)
            window = ctx->setValue("window", o3_new(cWindow1)());
        return window.toScr();
    }
    
    NSWindow* m_window;
    
    cWindow1()
    {
        m_window = [[NSApplication sharedApplication] mainWindow];
    }
    
    o3_begin_class(cWindow1Base)
    o3_end_class()
    
    o3_glue_gen()
    
    int x()
    {
        o3_trace3 trace;

        return m_window.frame.origin.x;
    }
    
    int setX(int x)
    {
        o3_trace3 trace;
        NSRect frame = m_window.frame;
        
        frame.origin.x = x;
        [m_window setFrame:frame display:NO];
        return x;
    }
    
    int y()
    {
        o3_trace3 trace;

        return m_window.frame.origin.y;
    }
    
    int setY(int y)
    {
        o3_trace3 trace;
        NSRect frame = m_window.frame;
        
        frame.origin.y = y;
        [m_window setFrame:frame display:NO];
        return y;
    }
    
    int width()
    {
        o3_trace3 trace;

        return m_window.frame.size.width;
    }
    
    int setWidth(int width)
    {
        o3_trace3 trace;
        NSRect frame = m_window.frame;
        
        frame.size.width = width;
        [m_window setFrame:frame display:YES];
        return width;
    }
    
    int height()
    {
        o3_trace3 trace;

        return m_window.frame.size.height;
    }
    
    int setHeight(int height)
    {
        o3_trace3 trace;
        NSRect frame = m_window.frame;
        
        frame.size.height = height;
        [m_window setFrame:frame display:YES];
        return height;
    }
};

}

#endif // O3_C_WINDOW1_APPLE_H
