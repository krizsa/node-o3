#ifndef O3_LIBAGG
#define O3_LIBAGG
namespace o3
{
	namespace agg
	{
#pragma warning( push)
#pragma warning( disable : 4100 4512 4244)
		#include "libagg-2.4/agg2d/agg2d.h"
		#include "libagg-2.4/agg2d/agg2d.cpp"
		#include "libagg-2.4/include/agg_rasterizer_scanline_aa.h"
		#include "libagg-2.4/include/agg_scanline_p.h"


		#include "libagg-2.4/src/agg_arc.cpp"
		#include "libagg-2.4/src/agg_arrowhead.cpp"
		#include "libagg-2.4/src/agg_bezier_arc.cpp"
		#include "libagg-2.4/src/agg_bspline.cpp"
		#include "libagg-2.4/src/agg_color_rgba.cpp"
		#include "libagg-2.4/src/agg_curves.cpp"
		#include "libagg-2.4/src/agg_embedded_raster_fonts.cpp"
		#include "libagg-2.4/src/agg_gsv_text.cpp"
		#include "libagg-2.4/src/agg_image_filters.cpp"
		#include "libagg-2.4/src/agg_line_aa_basics.cpp"
		#include "libagg-2.4/src/agg_line_profile_aa.cpp"
		#include "libagg-2.4/src/agg_rounded_rect.cpp"
		#include "libagg-2.4/src/agg_sqrt_tables.cpp"
		#include "libagg-2.4/src/agg_trans_affine.cpp"
		#include "libagg-2.4/src/agg_trans_double_path.cpp"
		#include "libagg-2.4/src/agg_trans_single_path.cpp"
		#include "libagg-2.4/src/agg_trans_warp_magnifier.cpp"
		#include "libagg-2.4/src/agg_vcgen_bspline.cpp"
		#include "libagg-2.4/src/agg_vcgen_contour.cpp"
		#include "libagg-2.4/src/agg_vcgen_dash.cpp"
		#include "libagg-2.4/src/agg_vcgen_markers_term.cpp"
		#include "libagg-2.4/src/agg_vcgen_smooth_poly1.cpp"
		#include "libagg-2.4/src/agg_vcgen_stroke.cpp"
		#include "libagg-2.4/src/agg_vpgen_clip_polygon.cpp"
		#include "libagg-2.4/src/agg_vpgen_clip_polyline.cpp"
		#include "libagg-2.4/src/agg_vpgen_segmentator.cpp"
#ifdef WINDOWS
		#include "libagg-2.4/font_win32_tt/agg_font_win32_tt.cpp"
#endif

#pragma warning(pop)

	};
};
#endif