#ifndef O3_LIBZBAR
#define O3_LIBZBAR

#include <stdio.h>

namespace o3
{
	namespace barcode
	{
#ifdef O3_LINUX
#define _snprintf snprintf
#define _strdup strdup
#define MAX_PATH 1024
#endif
		// since this is a third party lib anyway, I just turn off these warning in this file
		#pragma warning( disable : 4244)    // conversion without explicit cast
		#pragma warning( disable : 4127)    // conditional expression is constant
		#pragma warning( disable : 4018)    // signed/unsigned mismatch
		#pragma warning( disable : 4100)    // unreferenced formal parameter
		#pragma warning( disable : 4706)    // assignment within conditional expression
		typedef unsigned int DWORD;
		#define snprintf _snprintf   /* Added to v 1.2.19 */
		#define PRIx32 "lx"
		#define ENABLE_CODE128 1
		#define ENABLE_CODE39 1
		#define ENABLE_EAN 1
		#define ENABLE_I25 1
		#undef ENABLE_PDF417
		//#define ENABLE_QRCODE 1
		#undef HAVE_ATEXIT
		#undef HAVE_DLFCN_H
		#undef HAVE_FCNTL_H
		#undef HAVE_FEATURES_H
		#undef HAVE_GETPAGESIZE
		#undef HAVE_ICONV
		//#define HAVE_INTTYPES_H 1
		#undef HAVE_JPEGLIB_H
		#undef HAVE_LIBJPEG
		#undef HAVE_LIBPTHREAD
		#undef HAVE_LINUX_VIDEODEV2_H
		#undef HAVE_LINUX_VIDEODEV_H
		#undef HAVE_MEMORY_H

		#define HAVE_MEMSET 1
		#undef HAVE_MMAP
		#undef HAVE_POLL_H
		#undef HAVE_PTHREAD_H
		#undef HAVE_SETENV
		#define HAVE_STDINT_H 1
		#define HAVE_STDLIB_H 1
		#define HAVE_STRINGS_H 1
		#define HAVE_STRING_H 1
		#undef HAVE_SYS_IOCTL_H
		#undef HAVE_SYS_IPC_H
		#undef HAVE_SYS_MMAN_H
		#undef HAVE_SYS_SHM_H
		#define HAVE_SYS_STAT_H 1
		#define HAVE_SYS_TIMES_H 1
		//#define HAVE_SYS_TIME_H 1
		#define HAVE_SYS_TYPES_H 1
		#define HAVE_UINTPTR_T 1
		//#define HAVE_UNISTD_H 1
		//#undef HAVE_VFW_H
		#undef HAVE_X11_EXTENSIONS_XSHM_H
		#undef HAVE_X11_EXTENSIONS_XVLIB_H
		#undef ICONV_CONST
		#define LIB_VERSION_MAJOR 0
		#define LIB_VERSION_MINOR 2
		#define LIB_VERSION_REVISION 0
		#undef LT_OBJDIR
		//#undef NDEBUG
		//#undef NO_MINUS_C_MINUS_O
		//#undef PACKAGE "zbar"
		//#undef PACKAGE_BUGREPORT "spadix@users.sourceforge.net"
		//#undef PACKAGE_NAME "zbar"
		//#undef PACKAGE_STRING "zbar 0.10"
		//#undef PACKAGE_TARNAME "zbar"
		//#undef PACKAGE_VERSION "0.10"
		#define STDC_HEADERS 1
		//#undef VERSION "0.10"

		#define X_DISPLAY_MISSING 1
		#define ZBAR_VERSION_MAJOR 0
		#define ZBAR_VERSION_MINOR 10

		#include "zbar/include/zbar.h"

/*		#include "zbar/include/zbar/Decoder.h"
		#include "zbar/include/zbar/Exception.h"
		#include "zbar/include/zbar/Image.h"
		#include "zbar/include/zbar/ImageScanner.h"
		#include "zbar/include/zbar/Processor.h"
		#include "zbar/include/zbar/QZBar.h"
		#include "zbar/include/zbar/QZBarImage.h"
		#include "zbar/include/zbar/Scanner.h"
		#include "zbar/include/zbar/Symbol.h"
		#include "zbar/include/zbar/Video.h"
		#include "zbar/include/zbar/Window.h"
		#include "zbar/include/zbar/zbargtk.h"
		#include "zbar/include/zbar/ZBarImage.h"
		#include "zbar/include/zbar/ZBarImageScanner.h"
		#include "zbar/include/zbar/ZBarReaderController.h"
		#include "zbar/include/zbar/ZBarSymbol.h"
		#include "zbar/zbar/debug.h"
		#include "zbar/zbar/decoder.h"
		#include "zbar/zbar/error.h"
		#include "zbar/zbar/event.h"
		#include "zbar/zbar/image.h"
		#include "zbar/zbar/img_scanner.h"
		#include "zbar/zbar/mutex.h"
		#include "zbar/zbar/processor.h"
		#include "zbar/zbar/qrcode.h"
		#include "zbar/zbar/refcnt.h"
		#include "zbar/zbar/svg.h"
		#include "zbar/zbar/symbol.h"
		#include "zbar/zbar/thread.h"
		#include "zbar/zbar/timer.h"
		#include "zbar/zbar/video.h"
		#include "zbar/zbar/window.h"*/
		#include "zbar/zbar/config.c"
		#include "zbar/zbar/convert.c"
		#include "zbar/zbar/decoder.c"
		#include "zbar/zbar/error.c"
		#include "zbar/zbar/image.c"
		//#include "zbar/zbar/jpeg.c"
		//#include "zbar/zbar/libzbar.rc"
		//#include "zbar/zbar/Makefile.am.inc"
		
		#include "zbar/zbar/processor.c"
		#include "zbar/zbar/refcnt.c"
		#include "zbar/zbar/scanner.c"
		#include "zbar/zbar/symbol.c"
		#include "zbar/zbar/decoder/code128.h"
		#include "zbar/zbar/decoder/code39.h"
		#include "zbar/zbar/decoder/ean.h"
		#include "zbar/zbar/decoder/i25.h"
		#include "zbar/zbar/decoder/pdf417.h"
		#include "zbar/zbar/decoder/pdf417_hash.h"
		#include "zbar/zbar/decoder/qr_finder.h"
		#include "zbar/zbar/decoder/ean.c"

		#include "zbar/zbar/decoder/code128.c"
		#include "zbar/zbar/decoder/code39.c"
		#include "zbar/zbar/decoder/i25.c"
//		#include "zbar/zbar/decoder/pdf417.c"
//		#include "zbar/zbar/decoder/qr_finder.c"

//		#include "zbar/zbar/processor/posix.h"
//		#include "zbar/zbar/processor/lock.c"
//		#include "zbar/zbar/processor/null.c"
//		#include "zbar/zbar/processor/posix.c"
//		#include "zbar/zbar/processor/win.c"
//		#include "zbar/zbar/processor/x.c"

/*		#include "zbar/zbar/qrcode/bch15_5.h"
		#include "zbar/zbar/qrcode/binarize.h"
		#include "zbar/zbar/qrcode/isaac.h"
		#include "zbar/zbar/qrcode/qrdec.h"
		#include "zbar/zbar/qrcode/rs.h"
		#include "zbar/zbar/qrcode/util.h"
		#include "zbar/zbar/qrcode/bch15_5.c"
		#include "zbar/zbar/qrcode/binarize.c"
		#include "zbar/zbar/qrcode/isaac.c"
		#include "zbar/zbar/qrcode/qrdec.c"
		#include "zbar/zbar/qrcode/qrdectxt.c"
		#include "zbar/zbar/qrcode/rs.c"
		#include "zbar/zbar/qrcode/util.c"*/

		#include "zbar/zbar/img_scanner.c" // undefines CFG macro.. whatever it is for.. 

#pragma warning( default : 4244)    // conversion without explicit cast
#pragma warning( default : 4127)    // conditional expression is constant
#pragma warning( default : 4018)    // signed/unsigned mismatch
#pragma warning( default : 4100)    // unreferenced formal parameter
#pragma warning( default : 4706)    // assignment within conditional expression

	}
}

#endif //O3_LIBZBAR