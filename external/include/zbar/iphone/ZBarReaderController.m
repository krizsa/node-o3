//------------------------------------------------------------------------
//  Copyright 2009 (c) Jeff Brown <spadix@users.sourceforge.net>
//
//  This file is part of the ZBar Bar Code Reader.
//
//  The ZBar Bar Code Reader is free software; you can redistribute it
//  and/or modify it under the terms of the GNU Lesser Public License as
//  published by the Free Software Foundation; either version 2.1 of
//  the License, or (at your option) any later version.
//
//  The ZBar Bar Code Reader is distributed in the hope that it will be
//  useful, but WITHOUT ANY WARRANTY; without even the implied warranty
//  of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU Lesser Public License for more details.
//
//  You should have received a copy of the GNU Lesser Public License
//  along with the ZBar Bar Code Reader; if not, write to the Free
//  Software Foundation, Inc., 51 Franklin St, Fifth Floor,
//  Boston, MA  02110-1301  USA
//
//  http://sourceforge.net/projects/zbar
//------------------------------------------------------------------------

#import <sys/times.h>
#import <limits.h>

#import <zbar/ZBarReaderController.h>
#import "ZBarHelpController.h"
#import "debug.h"

#define MIN_QUALITY 10

NSString* const ZBarReaderControllerResults = @"ZBarReaderControllerResults";


@implementation ZBarReaderController

@synthesize scanner, readerDelegate, showsHelpOnFail;
@dynamic showsZBarControls;

- (id) init
{
    if(self = [super init]) {
        showsHelpOnFail = YES;
        hasOverlay = showsZBarControls =
            [self respondsToSelector: @selector(cameraOverlayView)];
        scanner = [ZBarImageScanner new];
        if([UIImagePickerController
               isSourceTypeAvailable: UIImagePickerControllerSourceTypeCamera])
            self.sourceType = UIImagePickerControllerSourceTypeCamera;
    }
    return(self);
}

- (void) initOverlay
{
    CGRect r = self.view.bounds;
    overlay = [[UIView alloc] initWithFrame: r];
    overlay.backgroundColor = [UIColor clearColor];

    controls = [UIView new];

    UIToolbar *toolbar = [UIToolbar new];
    toolbar.barStyle = UIBarStyleBlackOpaque;
    [toolbar sizeToFit];
    r.origin.y = r.size.height - 56;
    r.size.height = 56;
    controls.frame = r;
    r.origin.y = r.origin.x = 0;
    toolbar.frame = r;
    [controls addSubview: toolbar];
    [toolbar release];

    UIButton *btn = _ZBarButton(48, .1, .7, .1);
    btn.frame = CGRectMake(r.size.width / 4 + 4, 4,
                           r.size.width / 2 - 8, 48);
    [btn addTarget: self
         action: @selector(takePicture)
         forControlEvents: UIControlEventTouchDown];
    [btn setTitle: @"Scan!"
         forState: UIControlStateNormal];
    [controls addSubview: btn];

    btn = _ZBarButton(40, .85, .15, .15);
    btn.frame = CGRectMake(4, 4, r.size.width / 4 - 8, 48);
    [btn addTarget: self
         action: @selector(cancel)
         forControlEvents: UIControlEventTouchUpInside];
    [btn setTitle: @"done"
         forState: UIControlStateNormal];
    [controls addSubview: btn];

    controls.hidden = YES;
    [overlay addSubview: controls];
}

- (void) viewDidLoad
{
    [super viewDidLoad];
    [super setDelegate: self];
    if(hasOverlay)
        [self initOverlay];
}

- (void) cleanup
{
    [overlay release];
    overlay = nil;
    [controls release];
    controls = nil;
    [help release];
    help = nil;
}

- (void) viewDidUnload
{
    [self cleanup];
    [super viewDidUnload];
}

- (void) dealloc
{
    [self cleanup];
    [scanner release];
    scanner = nil;
    [super dealloc];
}

- (void) cancel
{
    [self performSelector: @selector(imagePickerControllerDidCancel:)
          withObject: self
          afterDelay: 0.1];
}

- (void) viewWillAppear: (BOOL) animated
{
    if(help) {
        [help.view removeFromSuperview];
        [help release];
        help = nil;
    }

    if(hasOverlay &&
       self.sourceType == UIImagePickerControllerSourceTypeCamera) {
        if(showsZBarControls || ![self cameraOverlayView])
            [self setCameraOverlayView: overlay];

        if(showsZBarControls)
            [self setShowsCameraControls: NO];
        else if([self cameraOverlayView] == overlay)
            [self setShowsCameraControls: YES];

        controls.hidden = !showsZBarControls;
    }
    [super viewWillAppear: animated];
}

- (void)  imagePickerController: (UIImagePickerController*) picker
  didFinishPickingMediaWithInfo: (NSDictionary*) info
{
    UIImage *img = [info objectForKey: UIImagePickerControllerOriginalImage];

    id results = [self scanImage: img];

    if(results) {
        NSMutableDictionary *newinfo = [info mutableCopy];
        [newinfo setObject: results
                 forKey: ZBarReaderControllerResults];
        SEL cb = @selector(imagePickerController:didFinishPickingMediaWithInfo:);
        if([readerDelegate respondsToSelector: cb])
            [readerDelegate imagePickerController: self
                            didFinishPickingMediaWithInfo: newinfo];
        else
            [self dismissModalViewControllerAnimated: YES];
        [newinfo release];
        return;
    }

    BOOL camera = (self.sourceType == UIImagePickerControllerSourceTypeCamera);
    BOOL retry = !camera || (hasOverlay && ![self showsCameraControls]);
    if(showsHelpOnFail && retry) {
        help = [ZBarHelpController new];
        help.delegate = self;

        if(camera) {
            help.wantsFullScreenLayout = YES;
            help.view.alpha = 0;
            [[self cameraOverlayView] addSubview: help.view];
            [UIView beginAnimations: @"ZBarHelp"
                    context: nil];
            help.view.alpha = 1;
            [UIView commitAnimations];
        }
        else
            [self presentModalViewController: help
                  animated: YES];
    }

    SEL cb = @selector(readerControllerDidFailToRead:withRetry:);
    if([readerDelegate respondsToSelector: cb])
        // assume delegate dismisses controller if necessary
        [readerDelegate readerControllerDidFailToRead: self
                        withRetry: retry];
    else if(!retry)
        // must dismiss stock controller
        [self dismissModalViewControllerAnimated: YES];
}

- (void) imagePickerControllerDidCancel: (UIImagePickerController*) picker
{
    SEL cb = @selector(imagePickerControllerDidCancel:);
    if([readerDelegate respondsToSelector: cb])
        [readerDelegate imagePickerControllerDidCancel: self];
    else
        [self dismissModalViewControllerAnimated: YES];
}

- (void) helpController: (ZBarHelpController*) hlp
   clickedButtonAtIndex: (NSInteger) idx
{
    if(self.sourceType == UIImagePickerControllerSourceTypeCamera) {
        [UIView beginAnimations: @"ZBarHelp"
                context: nil];
        hlp.view.alpha = 0;
        [UIView commitAnimations];
    }
    else
        [hlp dismissModalViewControllerAnimated: YES];
}

- (BOOL) showsZBarControls
{
    return(showsZBarControls);
}

- (void) setShowsZBarControls: (BOOL) show
{
    if(show && !hasOverlay)
        [NSException raise: NSInvalidArgumentException
            format: @"ZBarReaderController cannot set showsZBarControls=YES for OS<3.1"];

    showsZBarControls = show;
}

// intercept delegate as readerDelegate

- (void) setDelegate: (id <UINavigationControllerDelegate,
                           UIImagePickerControllerDelegate>) delegate
{
    self.readerDelegate = (id <ZBarReaderDelegate>)delegate;
}

- (id <NSFastEnumeration>) scanImage: (UIImage*) image
                                size: (CGSize) size
{
    timer_start;

    // limit the maximum number of scan passes
    int density;
    if(size.width > 480)
        density = (size.width / 240 + 1) / 2;
    else
        density = 1;
    [scanner setSymbology: 0
             config: ZBAR_CFG_X_DENSITY
             to: density];

    if(size.height > 480)
        density = (size.height / 240 + 1) / 2;
    else
        density = 1;
    [scanner setSymbology: 0
             config: ZBAR_CFG_Y_DENSITY
             to: density];

    ZBarImage *zimg = [[ZBarImage alloc]
                          initWithUIImage: image
                          size: size];
    int nsyms = [scanner scanImage: zimg];
    [zimg release];

    NSMutableArray *syms = nil;
    if(nsyms) {
        // quality/type filtering
        int max_quality = MIN_QUALITY;
        for(ZBarSymbol *sym in scanner.results) {
            zbar_symbol_type_t type = sym.type;
            int quality;
            if(type == ZBAR_QRCODE)
                quality = INT_MAX;
            else
                quality = sym.quality;

            if(quality < max_quality) {
                zlog(@"    type=%d quality=%d < %d\n",
                     type, quality, max_quality);
                continue;
            }

            if(max_quality < quality) {
                max_quality = quality;
                if(syms)
                    [syms removeAllObjects];
            }
            zlog(@"    type=%d quality=%d\n", type, quality);
            if(!syms)
                syms = [NSMutableArray arrayWithCapacity: 1];

            [syms addObject: sym];
        }
    }

    zlog(@"read %d filtered symbols in %gs total\n",
          (!syms) ? 0 : [syms count], timer_elapsed(t_start, timer_now()));
    return(syms);
}

- (id <NSFastEnumeration>) scanImage: (UIImage*) image
{
    CGSize size = image.size;
    if(size.width > 1280 || size.height > 1280) {
        size.width /= 2;
        size.height /= 2;
    }

    id <NSFastEnumeration> syms =
        [self scanImage: image
              size: size];
    if(!syms) {
        // make one more attempt for close up, grainy images
        size.width /= 2;
        size.height /= 2;
        syms = [self scanImage: image
                     size: size];
    }

    return(syms);
}

@end
