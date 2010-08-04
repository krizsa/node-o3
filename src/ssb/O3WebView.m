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
#import "O3WebView.h"
#import "O3URLProtocol.h"

@implementation O3WebView

- (id)initWithFrame:(NSRect)frame
{
    if (![super initWithFrame:frame frameName:@"O3" groupName:@"O3"])
        return nil;
	[NSURLProtocol registerClass:[O3URLProtocol class]];
    [WebView registerURLSchemeAsLocal:@"o3"];
	[self setMainFrameURL:@"o3:///bounce.html"];
    return self;
}

@end
