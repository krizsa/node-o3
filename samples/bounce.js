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

FPS = 30;

function bounce() {
	var screen, window, dx, dy, xmin, ymin, xmax, ymax;
    var o3 = document.getElementsByTagName("embed")[0];
    
	o3.loadModule("screen");
	o3.loadModule("window");
	screen = o3.screen;
	window = o3.window;
	window.width = 256;
	window.height = 256 + 24;
	dx = 8;
	dy = 6;
	xmin = 0;
	ymin = 0;
	xmax = screen.width - window.width;
	ymax = screen.height - window.height - 24;
	setInterval(function() {
		window.x += dx;
		window.y += dy;		
		if (window.x <= xmin || window.x >= xmax)
			dx = -dx;
		if (window.y <= ymin || window.y >= ymax)
			dy = -dy;
	}, 1000 / FPS);
}