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
#ifndef O3_C_WINDOW1_H
#define O3_C_WINDOW1_H

#ifdef O3_APPLE
#include "o3_cWindow1_apple.h"
#endif // O3_APPLE
#ifdef O3_WIN32
#include "shared/o3_glue.h"
#include "o3_cWindow1_win32.h"
#endif // O3_WIN32

#ifdef O3_WITH_GLUE
#ifdef O3_APPLE
#include "o3_scr_cWindow1_apple.h"
#endif // O3_APPLE
#ifdef O3_WIN32
#include "o3_scr_cWindow1_win32.h"
#endif // O3_WIN32

#endif // O3_WITH_GLUE 

#endif // O3_C_WINDOW1_H
