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
#ifndef O3_CONFIG_H
#define O3_CONFIG_H

#if defined(__APPLE__)
#define O3_APPLE
#elif defined(__linux__)
#define O3_LINUX
#elif defined(_WIN32)
#define O3_WIN32
#endif

#define O3_AUTO_CAPACITY    11
#define O3_TICK_SIZE        10
#define O3_CLS_TRAIT_COUNT  50
#define O3_EXT_TRAIT_COUNT  50
#define O3_VALUE_OFFSET     1000

#if defined(O3_APPLE)
#define O3_POSIX
#elif defined(O3_LINUX)
#define O3_POSIX
#elif defined(O3_WIN32)
#define _WIN32_DCOM
#define _CRT_SECURE_NO_WARNINGS
#endif

#define O3_LOG
#define O3_ASSERT

#if defined(O3_DEBUG)
#define O3_TRACE    0
#define O3_ASSERT
#define O3_LOG
#endif

#define O3_VERSION 0.9
#define O3_VERSION_STRING "v0_9"
#define O3_BASE_URL "http://www.ajax.org/o3test/"


#endif // O3_CONFIG_H
