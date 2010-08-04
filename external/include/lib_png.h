#ifndef O3_LIBPNG
#define O3_LIBPNG
namespace o3
{
	namespace png
	{
		using namespace ZLib;

		typedef void FAR   *voidpf;
		typedef void *voidp;
		typedef unsigned int uInt ;
		typedef unsigned char  Byte;  /* 8 bits */
		typedef Byte  FAR Bytef;

		#define png_snprintf _snprintf   /* Added to v 1.2.19 */
		#define png_snprintf2 _snprintf
		#define png_snprintf6 _snprintf


		#include "libpng/png.h"
		
		#include "libpng/pngpriv.h"



		#include "libpng/png.c"
		#include "libpng/pngerror.c"
		#include "libpng/pngget.c"
		#include "libpng/pngmem.c"
		#include "libpng/pngpread.c"
		#include "libpng/pngread.c"
		#include "libpng/pngrio.c"
		#include "libpng/pngrtran.c"
		#include "libpng/pngrutil.c"
		#include "libpng/pngset.c"
		#include "libpng/pngtest.c"
		#include "libpng/pngtrans.c"
		#include "libpng/pngwio.c"
		#include "libpng/pngwrite.c"
		#include "libpng/pngwtran.c"
		#include "libpng/pngwutil.c"

	}
}
#endif
