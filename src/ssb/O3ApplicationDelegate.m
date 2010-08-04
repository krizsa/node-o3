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
#import "O3ApplicationDelegate.h"
#import "O3WebView.h"

@implementation O3ApplicationDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)notification
{
	NSWindow *window = [NSWindow alloc];

	window = [window initWithContentRect:[NSScreen mainScreen].frame
							   styleMask:NSResizableWindowMask
								 backing:NSBackingStoreBuffered
								   defer:NO];
	[window setContentView:[[O3WebView alloc] initWithFrame:window.frame]];	
    [window makeKeyAndOrderFront:self];
}

@end
