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
#import "O3URLProtocol.h"

@implementation O3URLProtocol

+ (BOOL)canInitWithRequest:(NSURLRequest *)request
{
    return [[[request URL] scheme] isEqualToString:@"o3"];
}

+ (NSURLRequest *)canonicalRequestForRequest:(NSURLRequest *)request
{
    return request;
}

- (void)startLoading
{
    id<NSURLProtocolClient>	 client	= [self client];
	NSString				*path	= [[NSBundle mainBundle] resourcePath];
	NSURL					*URL	= [[self request] URL];
	NSData					*data;
	
    path = [path stringByAppendingString:URL.path];
	if (data = [NSData dataWithContentsOfFile:path]) {
        NSURLResponse *response = [[NSURLResponse alloc] initWithURL:URL
															MIMEType:nil
											   expectedContentLength:-1
													textEncodingName:nil];
        
		[client URLProtocol:self
		 didReceiveResponse:response
		 cacheStoragePolicy:NSURLCacheStorageNotAllowed];
        [client URLProtocol:self didLoadData:data];
        [client URLProtocolDidFinishLoading:self];
        [response release];
    } else {
		NSError *error = [NSError errorWithDomain:NSURLErrorDomain
											 code:NSURLErrorResourceUnavailable
										 userInfo:nil];
        
		[client URLProtocol:self didFailWithError:error];		
    }
}

- (void)stopLoading
{
}

@end
