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

#include <Cocoa/Cocoa.h>
#include <netdb.h>
#include <arpa/inet.h>
#undef assert
#undef TYPE_BOOL
#include <core/o3_core.h>

@interface O3Timer : NSObject {
	o3::iCtx *ctx;
}

@end

@implementation O3Timer

-(id)initWithCtx:(o3::iCtx *)aCtx
{
	if (![super init])
		return nil;
	ctx = aCtx;
	[self performSelector:@selector(timerDidFire:)
			   withObject:nil
               afterDelay:(NSTimeInterval) O3_TICK_SIZE / 1000];
	return self;
}

-(void)invalidate
{
	ctx = 0;
}

-(void)timerDidFire:(NSObject *)object
{
	if (ctx) {
		[self performSelector:@selector(timerDidFire:)
				   withObject:nil
				   afterDelay:(NSTimeInterval) O3_TICK_SIZE / 1000];
		ctx->loop()->wait(0);
	}
}

@end

#include "o3_np_plugin.h"
