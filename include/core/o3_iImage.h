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
#ifndef O3_I_IMAGE_H
#define O3_I_IMAGE_H

namespace o3 
{
	namespace Image
	{
		enum Modes
		{
			MODE_GRAY,
			MODE_BW,
			MODE_ARGB,
			MODE_RGB,
			__MODE_COUNT
		};
	};

	o3_iid(iImage, 0x4380b9f6, 
				0x9c2d, 
				0x48f6, 
				0xa3, 0xb7, 0x5, 0xff, 0x19, 0x8b, 0x7, 0x59);

	struct iImage : iUnk 
	{
		virtual size_t width() = 0;
		virtual size_t height() = 0;
		virtual size_t stride() = 0;
		virtual size_t bpp() = 0;
		virtual size_t mode_int() = 0;

		virtual unsigned char *getbufptr() = 0;
		virtual unsigned char *getrowptr(size_t y) = 0;
	};

};

#endif // O3_I_IMAGE_H