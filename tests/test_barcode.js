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

o3.loadModule("image");
o3.loadModule("console");
o3.loadModule("scanner");
o3.loadModule("barcode");
o3.print("modules loaded!\n");

var img = o3.image;

o3.print("blank image created?\n");

o3.print("attempting to load..\n");
img.src="barcode.png";
o3.print("attempting to scan codes..\n");
var thelist = img.scanbarcodes();
o3.print("found " + thelist.length + " codes: \n");
for (var i = 0;i<thelist.length;i++)
{
	o3.print(i+ ": "+thelist[i] + "\n");
}

o3.print("attempting to save..\n");
img.savePng("testbarcodeoutput.png");
""




