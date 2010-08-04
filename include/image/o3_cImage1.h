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
#ifndef O3_C_IMAGE1_H
#define O3_C_IMAGE1_H

#include <lib_zlib.h>
#include <core/o3_math.h>
#include <math.h>

#include <lib_png.h>
#include <lib_agg.h>


#ifndef __min
    #define __min(X,Y) (((X)<(Y))?(X):(Y))
#endif

#ifndef __max
    #define __max(X,Y) (((X)>(Y))?(X):(Y))
#endif
	
//#define IMAGE_ALPHAMAP_ENABLED

namespace o3 
{

	#include "o3_cImage1_CSS_colors.h"

	const double curve_distance_epsilon                  = 1e-30;
    const double curve_collinearity_epsilon              = 1e-30;
    const double curve_angle_tolerance_epsilon           = 0.01;

	
// png(bpp[1,8,24,32]) // return blob image.saveJpg(fsnode/filename, compression)
// jpg(compression) // return blob
// image = image.load(fsnode/filename); // autoformat image = image.loadJpg(fsnode/filename); image = image.loadPng(fsnode/filename);
	
// vector canvas API

	// clip();
// globalAlpha

	__inline double calc_sq_distance(double x1, double y1, double x2, double y2)
	{
		double dx = x2-x1;
		double dy = y2-y1;
		return dx * dx + dy * dy;
	}

	// custom io functions for the png lib:	
	void o3_read_data(png::png_structp png_ptr,
		png::png_bytep data, png::png_size_t length) 
	{
		siStream stream = (iStream*) png_ptr->io_ptr;
		if ( !stream || length != 
			stream->read((void*) data, (size_t) length )) 
		{				
			png::png_error(png_ptr, "PNG file read failed.");
		}
	}

	void o3_write_data(png::png_structp png_ptr,
		png::png_bytep data, png::png_size_t length)
	{
		siStream stream = (iStream*) png_ptr->io_ptr;
		if ( !stream || length != 
			stream->write((void*) data, (size_t) length )) 
		{				
			png::png_error(png_ptr, "PNG file write failed.");
		}
	}

	void o3_flush_data(png::png_structp png_ptr)
	{
		siStream stream = (iStream*) png_ptr->io_ptr;
		if ( !stream ) 
		{				
			png::png_error(png_ptr, "PNG file flush failed.");
		} 
		else
		{
			stream->flush();
		}
	}


	struct cImage1_Gradient: cScr 
	{
		o3_begin_class(cScr)
		o3_end_class()

		enum Types
		{
			GRADIENT_LIN,
			GRADIENT_RAD,
			__Type_Count
		};

		int m_type;
		V2<double> m_CP1;
		V2<double> m_CP2;
		tVec<unsigned int> m_colorstops;
	};

	struct cImage1 : cScr, iImage
	{
		o3_begin_class(cScr)
	        o3_add_iface(iImage)
		o3_end_class()

		o3_glue_gen()
		
		Str m_mode;
		int m_mode_int;
		bool m_graphics_attached;

		agg::Agg2D m_graphics;

		class Path 
		{
		public:
			tVec<V2<double> > m_path;
		};
		
		tVec<Path> m_paths;
		V2<double> m_lastpoint;
		
		class RenderState
		{
		public:
			unsigned int FillColor;
			unsigned int ClearColor;
			unsigned int StrokeColor;
			double StrokeWidth;

			int FillStyle;
			M33<double> Transformation;
			bool ClippingEnabled;

			V2<double> ClipTopLeft;
			V2<double> ClipBottomRight;

			// todo -> add clipping path!


			double miterLimit;
		};

		tVec<RenderState> m_renderstates;
		RenderState *m_currentrenderstate;

		size_t m_w, m_h, m_stride;
		int    m_bytesperpixel;
		int    m_bitdepth;
		Buf	   m_mem;
		Buf	   m_alphamem;

		cImage1()
		{
			m_w = m_h = m_stride = 0;
			m_bitdepth = 32;
			m_bytesperpixel = 4;
			m_graphics_attached = false;
			m_mode = Str("argb");
			SetupRenderState();
			Ensure32BitSurface();
		};

		void SetupMode(size_t w, size_t h, const Str &mode)
		{
			m_w = w;
			m_h = h;
			m_stride = (m_w+7)&~(7);
			m_graphics_attached = false;
			m_mode = mode;
			m_bytesperpixel = 4;
			m_bitdepth = 32;
			if (m_mode == "argb")
			{
				m_bytesperpixel = 4;
				m_bitdepth = 32;
				m_mode_int = Image::MODE_ARGB;
			}
			else if (m_mode == "gray" || m_mode == "grey")
			{
				m_bitdepth = 8;
				m_bytesperpixel= 1;
				m_mode_int = Image::MODE_GRAY;
			}
			else if (m_mode == "bw")
			{
				m_bitdepth = 1;
				m_bytesperpixel = 1;
				m_mode_int = Image::MODE_BW;

			}
			else if (m_mode == "rgb")
			{
				m_bitdepth = 24;
				m_bytesperpixel = 3;
				m_mode_int = Image::MODE_RGB;
			}

			SetupBuffer();
			SetupRenderState();

			if (m_mode_int == Image::MODE_ARGB)
			{
				Ensure32BitSurface();
			};
		};
		cImage1(size_t w, size_t h, const Str &mode)
		{
			SetupMode(w,h,mode);

		};

		void SetupBuffer()
		{	
			size_t newsize = 0;
			switch(m_mode_int)
			{
			case Image::MODE_BW:
				newsize = (m_stride * m_h)/8;
				break;
			default:
				newsize = m_stride * m_h * m_bytesperpixel;
				break;
			}
			m_mem.resize(newsize);
			m_graphics_attached = false;
		};

		static o3_ext("cO3") o3_fun siScr image()
		{
			o3_trace3 trace;
			return o3_new(cImage1)();
		}

		static o3_ext("cO3") o3_fun siScr image(size_t w, size_t h, const char* mode = "argb" )
		{
			return o3_new(cImage1)(w,h,mode);
		}

		o3_get Str mode(){return m_mode;}

		o3_get size_t x(){return m_w;}
		o3_get size_t y(){return m_h;}

		virtual o3_get size_t width(){return m_w;}
		virtual o3_get size_t height(){return m_h;}

		virtual size_t stride(){return m_stride;};
		virtual size_t bpp(){return m_bitdepth;};
		virtual size_t mode_int(){return m_mode_int;}

		virtual unsigned char *getbufptr(){return (unsigned char*)m_mem.ptr();};
		virtual unsigned char *getrowptr(size_t y){return _getRowPtr(y);};

		__inline unsigned char *_getRowPtr(size_t y)
		{
			if (m_mode_int == Image::MODE_BW) 
			{
				if (y< (int)m_h) 
				{
					return ((unsigned char *)m_mem.ptr() + (m_stride * y) / 8);
				};
			}
			else
			{
				if (y< m_h) 
				{
					return ((unsigned char *)m_mem.ptr() + (m_stride * y) * m_bytesperpixel);
				}
			};
			return 0;
		};
		
		o3_fun void clear(int signed_color)
		{
			unsigned int color = (unsigned int) signed_color;
			switch(m_mode_int)
			{
			case Image::MODE_ARGB:
				m_mem.set<int>(0,color, m_mem.size());
				break;
			case Image::MODE_BW:
				if (color &0xffffff)
				{
					m_mem.set<unsigned char>(0,0xff, m_mem.size());
				}
				else
				{
					m_mem.set<unsigned char>(0,0, m_mem.size());
				}

				break;

			default:
				for (size_t y = 0;y<m_h;y++)
				{
					for (size_t x=0;x<m_w;x++) 
					{
						setPixel(x,y,color);
					};
				};
				break;
			};
		};

		o3_fun void setPixel(size_t x, size_t y, int signed_color)
		{
			unsigned int color = (unsigned int) signed_color;
			unsigned char *D = _getRowPtr(y);
			if(D)
			{
				if (x >= 0 && x < m_w)
				{
					switch(m_mode_int)
					{
						
					case Image::MODE_BW:
						{
							int shift = x&7;
							x>>=3;
							int mask = 1<<(7-shift);
							unsigned char *pixeldest = D + x;
							if (color&0xffffff)
							{
								*pixeldest |= mask;
							}
							else
							{
								*pixeldest &= ~mask;
							}
						};break;
					case Image::MODE_GRAY:
						{
							unsigned char *pixeldest = D+x;
							unsigned char *srcchannels = (unsigned char *) &color;
							unsigned char a = srcchannels[3];
							if (a == 255)
							{
								*pixeldest = srcchannels[0];
							}
							else
							{
								unsigned char *dstchannels = (unsigned char *) pixeldest;
								unsigned char inva = ~a;
								
								srcchannels[0]= (dstchannels[0]*inva + srcchannels[0]*a)>>8;
								*pixeldest = srcchannels[0];
							}

						}break;
					case Image::MODE_ARGB:
						{
							unsigned int *pixeldest = ((unsigned int *)(D)) + x;
							
							unsigned char *srcchannels = (unsigned char *) &color;
							unsigned char a = srcchannels[3];
							if (a == 255)
							{
								*pixeldest = color;
							}
							else
							{
								unsigned char *dstchannels = (unsigned char *) pixeldest;
								unsigned char inva = ~a;
								srcchannels[3]= 0xff; //TODO dst alpha needs to get some meaning!
								for (int j= 0;j<3;j++)
								{
									srcchannels[j]= (dstchannels[j]*inva + srcchannels[j]*a)>>8;
								}
								//TODO: add blendpixel stuff that properly uses the alpha information.

								*pixeldest = color;
							};
						};break;
					};
				};
			};
		};

		o3_fun int getPixel(size_t x, size_t y)
		{
			unsigned char *D = _getRowPtr(y);
			if(D)
			{
				if (x >= 0 && x < m_w)
				{
					switch (m_mode_int)
					{
					case Image::MODE_BW:
						{
							int shift = x&7;
							x>>=3;
							int mask = 1<<(7-shift);
							unsigned char *pixeldest = D + x;
							if (*pixeldest & mask) return 0xffffffff;
							return 0xff000000;
						};break;

					case Image::MODE_ARGB:
						{
							unsigned int *pixeldest = ((unsigned int *)(D)) + x;
							return *pixeldest;
						};break;
					};
				};
			};
			return 0;
		};

		o3_set siFs src(iFs* file, siEx* ex=0)
		{
			using namespace png;			

			// unable to open
			if (!file || !file->exists()) 
			{
				cEx::fmt(ex,"Invalid file."); 
				return file;
			}

			siStream stream = file->open("r", ex);
			if (!stream)				
				return file;

			// create read struct
			png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);

			// check pointer
			if (png_ptr == 0)
			{
				cEx::fmt(ex,"Creating PNG read struct failed."); 
				return file;
			}

			// create info struct
			png_infop info_ptr = png_create_info_struct(png_ptr);

			// check pointer
			if (info_ptr == 0)
			{
				png_destroy_read_struct(&png_ptr, 0, 0);
				cEx::fmt(ex,"Creating PNG info struct failed."); 
				return file;
			}

			// set error handling
			if (setjmp(png_jmpbuf(png_ptr)))
			{
				png_destroy_read_struct(&png_ptr, &info_ptr, 0);
				cEx::fmt(ex,"Setting up PNG error handling failed."); 				
				return file;
			}

			// I/O initialization using custom o3 methods
			png_set_read_fn(png_ptr,(void*) stream.ptr(), (png_rw_ptr) &o3_read_data);
			
			// read entire image (high level)
			png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_EXPAND, 0);

			// convert the png bytes to BGRA			
			int W = png_get_image_width(png_ptr, info_ptr);
			int H = png_get_image_height(png_ptr, info_ptr);

			// get color information
			int color_type = png_get_color_type(png_ptr, info_ptr);

			png_bytep* row_pointers = png_get_rows(png_ptr, info_ptr);

			switch(color_type)
			{ 
				case PNG_COLOR_TYPE_RGB:
					{
						SetupMode(W,H, "argb");

//						int pos = 0;

						// get color values
						for(int i = 0; i < (int) m_h; i++)
						{
							unsigned char *D = getrowptr(i);
							for(int j = 0; j < (int)(3 * m_w); j += 3)
							{
								*D++ = row_pointers[i][j + 2];	// blue
								*D++ = row_pointers[i][j + 1];	// green
								*D++ = row_pointers[i][j];		// red
								*D++ = 0xff;						// alpha
							}
						}

					};break;
				case PNG_COLOR_TYPE_RGB_ALPHA:
					{
						SetupMode(W,H, "argb");
					};break;
					
				case PNG_COLOR_TYPE_GRAY:
					{
						SetupMode(W,H, "gray");
					};break;
				case PNG_COLOR_TYPE_GRAY_ALPHA:
					{
						SetupMode(W,H, "argb");
					};break;
				break;

				default:
					png_destroy_read_struct(&png_ptr, &info_ptr, 0);
					cEx::fmt(ex,"PNG unsupported color type.");
					return file;
			}

			png_destroy_read_struct(&png_ptr, &info_ptr, 0);			
			return file;
		};
		
		o3_fun int savePng(iFs* file, siEx* ex = 0)
		{
			using namespace png;
			png_structp png_ptr;
			png_infop info_ptr;
			
			if (m_w==0 ||m_h == 0)
			{
				cEx::fmt(ex,"[write_png_file] image must have both width and height >0 before something can be written!");			
				return 0;
			}

			/* create file */
			if (!file) 
			{
				cEx::fmt(ex,"Invalid file."); 
				return 0;
			}

			siStream stream = file->open("w", ex);
			if (!stream)				
				return 0;

			/* initialize stuff */
			png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

			if (!png_ptr)
			{
				cEx::fmt(ex,"[write_png_file] png_create_write_struct failed");
				return 0;
			}

			info_ptr = png_create_info_struct(png_ptr);
			
			if (!info_ptr)
			{
				cEx::fmt(ex,"[write_png_file] png_create_info_struct failed");
				return 0; 
			}
			
			if (setjmp(png_jmpbuf(png_ptr)))
			{
				cEx::fmt(ex,"[write_png_file] Error during init_io");
				return 0;
			}

			png_set_write_fn(png_ptr,(void*) stream.ptr(), 
				(png_rw_ptr) &o3_write_data, (png_flush_ptr) &o3_flush_data);


			/* write header */
			if (setjmp(png_jmpbuf(png_ptr)))
			{
				cEx::fmt(ex,"[write_png_file] Error during writing header");
				return 0;
			};

			int color_type = 0;
			int bitdepth = 8;
			switch (m_mode_int )
			{
			
			case Image::MODE_BW: 
				color_type = PNG_COLOR_TYPE_GRAY; 
				bitdepth = 1;
				break;
			case Image::MODE_GRAY: 
				color_type = PNG_COLOR_TYPE_GRAY; 
				break;
			default: 
				color_type = PNG_COLOR_TYPE_RGB_ALPHA;
				break;
			}
			// TODO! add 1bit save
			
			png_set_IHDR(png_ptr, info_ptr, m_w, m_h,
				bitdepth, color_type, PNG_INTERLACE_NONE,
				PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

			png_write_info(png_ptr, info_ptr);

			if (setjmp(png_jmpbuf(png_ptr)))
			{
				cEx::fmt(ex,"[write_png_file] Error during writing bytes");
				return 0;
			};

			tVec<png_bytep> row_pointers(m_h);
			switch (m_mode_int)
			{
				case Image::MODE_ARGB:
				{
					tVec <unsigned int> row(m_w);
					for (size_t y = 0;y<m_h;y++)
					{
						unsigned int *D = (unsigned int *)_getRowPtr(y);
						for (size_t i =0 ;i<m_w;i++)
						{
							unsigned int const pixel = *D++;
							unsigned int shuffled = ((pixel >> 24)&0x255) + ((pixel << 8)&0xffffff00);
							row[i] = shuffled;

							unsigned char *c = (unsigned char*)&row[i];
							c[0] = (unsigned char)(pixel>>16);
							c[1] = (unsigned char)(pixel>>8);
							c[2] = (unsigned char)(pixel);
							c[3] = (unsigned char)(pixel>>24);
						}
						png_write_row(png_ptr, (png_bytep)row.ptr());
					};
				}
				break;
				case Image::MODE_RGB:
				{
					tVec <unsigned int> row(m_w);
					for (size_t y = 0;y<m_h;y++)
					{
						unsigned char *D = (unsigned char *)_getRowPtr(y);
						for (size_t i =0 ;i<m_w;i++)
						{
							unsigned char R = *D++;
							unsigned char G = *D++;
							unsigned char B = *D++;
							unsigned int const pixel = (R << 24) + (G << 16) + (B << 8) + 0xff ;
							row[i] = pixel;
						}
						png_write_row(png_ptr, (png_bytep)row.ptr());
					};
				}
				break;
				case Image::MODE_GRAY:
				{
					tVec <unsigned char> row(m_w);
					for (size_t y = 0;y<m_h;y++)
					{
						unsigned char *D = (unsigned char *)_getRowPtr(y);
						png_write_row(png_ptr, D);
					};
				}
				break;
				case Image::MODE_BW:
				{
					tVec <unsigned int> row(m_w);
					for (size_t y = 0;y<m_h;y++)
					{
						unsigned char *D = (unsigned char *)_getRowPtr(y);
						png_write_row(png_ptr, D);
					};
				}
				break;

			};

			//png_write_image(png_ptr, row_pointers.ptr());


			if (setjmp(png_jmpbuf(png_ptr)))
			{
				cEx::fmt(ex,"[write_png_file] Error during end of write");
				return 0;
			};

			png_write_end(png_ptr, NULL);

			/* cleanup heap allocation */
			
			return 1;
		};
	
		void Ensure32BitSurface()
		{
			if (m_mode_int != Image::MODE_ARGB)
			{
// TODO -- convert existing bitmap to 32 bit and remember old mode. 
			};

			if (!m_graphics_attached && m_mode_int == Image::MODE_ARGB) 
			{
				m_graphics.attach((unsigned char *)m_mem.ptr(), m_w, m_h, m_stride*4);
// TODO -- check different pixel alignments
//				m_graphics.viewport(0,0,m_w, m_h, 0,0,m_w, m_h, agg::Agg2D::ViewportOption::XMidYMid);
				RestoreStateToGraphicsObject();
				m_graphics_attached = true;
			};
		};

		void AttachAlpha()
		{
			if (m_alphamem.size() < m_stride*m_h)
			{
				m_alphamem.resize(m_stride*m_h);
				m_alphamem.set<unsigned char>(0, 0, m_alphamem.size());
			};
			m_graphics.attachalpha((unsigned char *)m_alphamem.ptr(), m_w, m_h, m_stride);
		};

		o3_fun void rect(int x, int y, int w, int h, int signed_color)    // !ALLMODES!
		{
			unsigned int color = (unsigned int) signed_color;
			switch (m_mode_int)
			{
			case Image::MODE_ARGB:
				{
					int x1 = __max(0,x);
					int x2 = __min(x+w, (int)m_w);
					int actualw = x2-x1;
					if (actualw <= 0 ) return;
					int y1 = __max(0, y);
					int y2 = __min(y+h, (int)m_h);
					for (int sy = y1;sy<y2;sy++)
					{
						unsigned char *S = _getRowPtr(sy);
						m_mem.set<unsigned int>((int)(S-(unsigned char*)m_mem.ptr())+x1*sizeof(unsigned int), color, actualw*sizeof(unsigned int));
					};
				};
				break;
			default:
				for (int sy = y;sy<y+h;sy++)
				{
					for (int sx = x;sx<x+w;sx++)
					{
						setPixel(sx,sy,color);
					};
				};
			}
		};

		o3_fun void line(int x0,int y0,int x1,int y1,int signed_color)    // !ALLMODES!
		{
			unsigned int color = (unsigned int) signed_color;
			bool steep = (abs(y1 - y0) > abs(x1 - x0));
			if (steep)
			{			 
				swap(x0, y0);
				swap(x1, y1);
			}
			if (x0 > x1)
			{
				swap(x0, x1);
				swap(y0, y1);
			}
			int deltax = x1 - x0;
			int deltay = abs(y1 - y0);
			int error = deltax / 2;
			int ystep;
			int y = y0;
			if (y0 < y1) 
			{
				ystep = 1;
			}
			else 
			{
				ystep = -1;
			};

			 for (int x=x0;x<x1;x++)
			 {
				 if (steep)
				 {
					 setPixel(y,x, color);
				 }
				 else 
				 {
					 setPixel(x,y, color);
				 }
				 error = error - deltay;
				 if( error < 0) 
				 {
					 y = y + ystep;
					 error = error + deltax;
				 }

			 }
		};

		o3_fun int decodeColor(const Str &style)
		{
			unsigned int stylesize = style.size();
			if (stylesize>0)
			{
				if (style[0] == '#')
				{
					int HexDigitsFound = 0;
					int digits[6];
					char *d = (char * ) (style.ptr()) + 1;
					for (;*d;d++)
					{
						switch (*d)
						{
						case '0':
						case '1':
						case '2':
						case '3':
						case '4':
						case '5':
						case '6':
						case '7':
						case '8':
						case '9':
							digits[HexDigitsFound++] = *d-'0';
							break;
						case 'a':
						case 'b':
						case 'c':
						case 'd':
						case 'e':
						case 'f':
							digits[HexDigitsFound++] = (*d-'a')+10;
							break;
						case 'A':
						case 'B':
						case 'C':
						case 'D':
						case 'E':
						case 'F':
							digits[HexDigitsFound++] = (*d-'A')+10;
							break;
						};
						if (HexDigitsFound==6)
						{
							unsigned char Res[3] = 
							{
								(unsigned char)((digits[0]<<4)+ digits[1]),
								(unsigned char)((digits[2]<<4)+ digits[3]),
								(unsigned char)((digits[4]<<4)+ digits[5])
							};
							
							return 0xff000000 + (Res[0]<<16) + (Res[1]<<8) + Res[2];
						};
					}

					if (HexDigitsFound > 2 )
					{
						unsigned char Res[3] = 
						{
							(unsigned char)((digits[0]<<4)+ digits[0]),
							(unsigned char)((digits[1]<<4)+ digits[1]),
							(unsigned char)((digits[2]<<4)+ digits[2])
						};
						return 0xff000000 + (Res[0]<<16) + (Res[1]<<8) + Res[2];
					};

					return 0xff000000;					
					
				}
				unsigned int color = 0;
				unsigned int index = 0;
				while (color < 0x0f000000)
				{

					if (index > stylesize) 
					{
						color = 0xff000000;
					}
					else
					{
						int c;
						if (index == stylesize)	
						{
							c = 'q'-'a';
						}
						else
						{
							c = style[index];
							if (c>='a' && c<='z')
							{
								c-='a';
							}
							else
							{
								if (c>='A' && c<='Z')
								{
									c-='A';
								}
								else
								{
									c='q'-'a';
								}
							}
						};
						index++;
						color = css_lut[color][c];
					};
				}
				unsigned int masked = color&0xff000000;
				
				if (masked == 0xff000000)
				{
					return color;
				};
				
				int totalchannels;

				double AlphaRes=0;

				if (masked == 0x2f000000)
				{
					totalchannels = 4;
				}
				else
				{
					totalchannels = 3;
					AlphaRes = 1;
				}

				int Res[3]={0,0,0};
				int current = 0;
				
//				int val = 0;
				bool afterdot = false;
				int digitsafterdot = 0;
				char *d = (char * ) (style.ptr());
				for (;*d;d++)
				{
					if (*d>='0' && *d<='9') 
					{
						if (current<3)
						{
							Res[current] *= 10;
							Res[current] += (*d-'0');
						}
						else
						{
							AlphaRes *= 10;
							AlphaRes += (*d-'0');
							if (afterdot) digitsafterdot++;
						}
					}
					else
					{
						if (*d ==',')
						{
							current ++;
							if (current == totalchannels)break;
							
						}
						if (current == 3 && *d == '.')
						{
							afterdot = true;
						}
					}
				}
				while (digitsafterdot > 0) 
				{ 
					AlphaRes /= 10;digitsafterdot--;
				};
				color = (Res[0]<<16) + (Res[1]<<8) + Res[2] + ((int)(AlphaRes*255.0f)<<24);

				return color;

			}
			else
			{
				return 0xff000000;
			};
		};

		o3_set void fillStyle(const Str &style)
		{
			m_currentrenderstate->FillColor = decodeColor(style);

			unsigned int color =  m_currentrenderstate->FillColor;
			unsigned char *c = (unsigned char *)&color;
			m_graphics.fillColor(c[2], c[1], c[0], c[3]);
		};

		o3_set void strokeStyle(const Str &style)
		{
			m_currentrenderstate->StrokeColor = decodeColor(style);

			unsigned int color =  m_currentrenderstate->StrokeColor;
			unsigned char *c = (unsigned char *)&color;
			m_graphics.lineColor(c[2], c[1], c[0], c[3]);
		};

		o3_set void strokeWidth (double Width)
		{
			m_currentrenderstate->StrokeWidth = Width;
		};

		o3_fun void clearRect(double xx, double yy, double ww, double hh)
		{
			Ensure32BitSurface();
			unsigned int color =  m_currentrenderstate->ClearColor;
			unsigned char *c = (unsigned char *)&color;

			m_graphics.resetPath();

			V2<double> p1(xx,yy);
			V2<double> p2(xx+ww,yy);
			V2<double> p3(xx+ww,yy+hh);
			V2<double> p4(xx,yy+hh);
			
			p1  = m_currentrenderstate->Transformation.Multiply(p1);
			p2  = m_currentrenderstate->Transformation.Multiply(p2);
			p3  = m_currentrenderstate->Transformation.Multiply(p3);
			p4  = m_currentrenderstate->Transformation.Multiply(p4);
			
			m_graphics.moveTo(p1.x,p1.y);
			m_graphics.lineTo(p2.x,p2.y);
			m_graphics.lineTo(p3.x,p3.y);
			m_graphics.lineTo(p4.x,p4.y);

			m_graphics.closePolygon();
			m_graphics.fillColor(c[2], c[1], c[0], c[3]);
			m_graphics.drawPath(agg::Agg2D::FillOnly);
			{

				unsigned int color =  m_currentrenderstate->FillColor;
				unsigned char *c = (unsigned char *)&color;

				m_graphics.fillColor(c[2], c[1], c[0], c[3]);
			};
		};

		o3_fun void fillRect(double xx, double yy, double ww, double hh)
		{
			Ensure32BitSurface();

			m_graphics.resetPath();

			V2<double> p1(xx,yy);
			V2<double> p2(xx+ww,yy);
			V2<double> p3(xx+ww,yy+hh);
			V2<double> p4(xx,yy+hh);
			
			p1  = m_currentrenderstate->Transformation.Multiply(p1);
			p2  = m_currentrenderstate->Transformation.Multiply(p2);
			p3  = m_currentrenderstate->Transformation.Multiply(p3);
			p4  = m_currentrenderstate->Transformation.Multiply(p4);
			
			m_graphics.moveTo(p1.x,p1.y);
			m_graphics.lineTo(p2.x,p2.y);
			m_graphics.lineTo(p3.x,p3.y);
			m_graphics.lineTo(p4.x,p4.y);
			m_graphics.closePolygon();

			m_graphics.drawPath(agg::Agg2D::FillOnly);
		};

		o3_fun void strokeRect(double xx, double yy, double ww, double hh)
		{
			Ensure32BitSurface();

			m_graphics.resetPath();

			V2<double> p1(xx,yy);
			V2<double> p2(xx+ww,yy);
			V2<double> p3(xx+ww,yy+hh);
			V2<double> p4(xx,yy+hh);
			
			p1  = m_currentrenderstate->Transformation.Multiply(p1);
			p2  = m_currentrenderstate->Transformation.Multiply(p2);
			p3  = m_currentrenderstate->Transformation.Multiply(p3);
			p4  = m_currentrenderstate->Transformation.Multiply(p4);
			
			m_graphics.moveTo(p1.x,p1.y);
			m_graphics.lineTo(p2.x,p2.y);
			m_graphics.lineTo(p3.x,p3.y);
			m_graphics.lineTo(p4.x,p4.y);

			m_graphics.closePolygon();

			m_graphics.drawPath(agg::Agg2D::StrokeOnly);
		};

		void SetupRenderState()
		{
			RenderState RS;
			RS.ClipTopLeft.x = 0;
			RS.ClipTopLeft.y = 0;
			RS.ClipBottomRight.x = m_w;
			RS.ClipBottomRight.y = m_h;

			RS.ClearColor = 0xffffffff;
			RS.StrokeWidth = 1;
			RS.ClippingEnabled = false;

			m_renderstates.push(RS);
			m_currentrenderstate = &m_renderstates[m_renderstates.size()-1];

			strokeStyle("black");
			fillStyle("black");

		};

		o3_fun void moveTo(double x, double y)
		{
			m_paths.push(Path());
			V2<double> point(x,y);
			point = TransformPoint(point);
			m_paths[m_paths.size()-1].m_path.push(point);
			m_lastpoint = point;
		}

		o3_fun void lineTo(double x, double y)
		{
			if (m_paths.size() == 0)
			{
				m_paths.push(Path());
				m_paths[m_paths.size()-1].m_path.push(m_lastpoint);
			};
			V2<double> point(x,y);
			m_paths[m_paths.size()-1].m_path.push(TransformPoint(point));
			m_lastpoint.x = x;
			m_lastpoint.y = y;
		};
		
		o3_fun void closePath()
		{
			if (m_paths.size() == 0) return;
			if (m_paths[0].m_path.size()<2) return;

			V2<double> first;
			first.x = m_paths[m_paths.size()-1].m_path[0].x;
			first.y = m_paths[m_paths.size()-1].m_path[0].y;

			m_paths[m_paths.size()-1].m_path.push(first);
			m_lastpoint = first;
		}

		o3_fun void beginPath()
		{
			m_paths.clear();
		}

		o3_fun void fill()
		{
			Ensure32BitSurface();
			m_graphics.resetPath();

//			TransformCurrentPath();

			for (size_t i =0 ;i<m_paths.size();i++)
			{
				if (m_paths[i].m_path.size()>1)
				{
					V2<double> Prev = m_paths[i].m_path[0];
					m_graphics.moveTo(Prev.x, Prev.y);
					for (size_t j = 1;j<m_paths[i].m_path.size();j++)
					{
						V2<double> Cur;
						Cur.x = m_paths[i].m_path[j].x;
						Cur.y = m_paths[i].m_path[j].y;
						m_graphics.lineTo(Cur.x, Cur.y);
//						line(Prev.x, Prev.y, Cur.x, Cur.y, color);
						Prev.x = Cur.x;
						Prev.y = Cur.y;
					};
					m_graphics.closePolygon();
				};
			};
			m_graphics.drawPath(agg::Agg2D::FillOnly);
			//m_paths.clear();


		};

		o3_fun void stroke()
		{
			Ensure32BitSurface();
			m_graphics.resetPath();
			m_graphics.lineWidth(m_currentrenderstate->StrokeWidth);
//			m_graphics.line(0,0,m_w, m_h);

//			TransformCurrentPath();

			for (size_t i =0 ;i<m_paths.size();i++)
			{
				if (m_paths[i].m_path.size()>1)
				{
					V2<double> Prev = m_paths[i].m_path[0];
					m_graphics.moveTo(Prev.x, Prev.y);
					for (size_t j = 1;j<m_paths[i].m_path.size();j++)
					{
						V2<double> Cur;
						Cur.x = m_paths[i].m_path[j].x;
						Cur.y = m_paths[i].m_path[j].y;
						m_graphics.lineTo(Cur.x, Cur.y);
//						line(Prev.x, Prev.y, Cur.x, Cur.y, color);
						Prev.x = Cur.x;
						Prev.y = Cur.y;
					};
				};
			};
			m_graphics.drawPath(agg::Agg2D::StrokeOnly);
//			m_paths.clear();
		};

		enum curve_recursion_limit_e { curve_recursion_limit = 32 };

		class QuadraticCurveGen
		{
		public:
			

			QuadraticCurveGen() : 
				m_approximation_scale(1.0),
				m_angle_tolerance(0.0),
				m_count(0)
			{
			}

			QuadraticCurveGen(double x1, double y1, double x2, double y2, double x3, double y3) :
				m_approximation_scale(1.0),
				m_angle_tolerance(0.0),
				m_count(0)
			{ 
				init(x1, y1, x2, y2, x3, y3);
			}

			
			void init(double x1, double y1, double x2, double y2, double x3, double y3)
			{
				m_points.clear();
				m_distance_tolerance_square = 0.5 / m_approximation_scale;
				m_distance_tolerance_square *= m_distance_tolerance_square;
				bezier(x1, y1, x2, y2, x3, y3);
				m_count = 0;
			};

			unsigned vertex(double* x, double* y)
			{
				if(m_count >= m_points.size()) return agg::agg::path_cmd_stop;
				V2<double> &p = m_points[m_count++];
				*x = p.x;
				*y = p.y;
				return (m_count == 1) ? agg::agg::path_cmd_move_to : agg::agg::path_cmd_line_to;
			}

		private:
			void recursive_bezier(double x1, double y1, double x2, double y2, double x3, double y3,unsigned level)
			{
				if(level > curve_recursion_limit) 
				{
					return;
				}

				// Calculate all the mid-points of the line segments
				//----------------------
				double x12   = (x1 + x2) / 2;                
				double y12   = (y1 + y2) / 2;
				double x23   = (x2 + x3) / 2;
				double y23   = (y2 + y3) / 2;
				double x123  = (x12 + x23) / 2;
				double y123  = (y12 + y23) / 2;

				double dx = x3-x1;
				double dy = y3-y1;
				double d = fabs(((x2 - x3) * dy - (y2 - y3) * dx));
				double da;

				if(d > curve_collinearity_epsilon)
				{ 
					// Regular case
					if(d * d <= m_distance_tolerance_square * (dx*dx + dy*dy))
					{
						// If the curvature doesn't exceed the distance_tolerance value
						// we tend to finish subdivisions.
						if(m_angle_tolerance < curve_angle_tolerance_epsilon)
						{
							m_points.push(V2<double>(x123, y123));
							return;
						}

						// Angle & Cusp Condition
						//----------------------
						da = fabs(atan2(y3 - y2, x3 - x2) - atan2(y2 - y1, x2 - x1));
						if(da >= pi) da = 2*pi - da;

						if(da < m_angle_tolerance)
						{
							// Finally we can stop the recursion
							m_points.push(V2<double>(x123, y123));
							return;                 
						}
					}
				}
				else
				{
					// Collinear case
					da = dx*dx + dy*dy;
					if(da == 0)
					{
						d = calc_sq_distance(x1, y1, x2, y2);
					}
					else
					{
						d = ((x2 - x1)*dx + (y2 - y1)*dy) / da;
						if(d > 0 && d < 1)
						{
							// Simple collinear case, 1---2---3
							// We can leave just two endpoints
							return;
						}
							 if(d <= 0) d = calc_sq_distance(x2, y2, x1, y1);
						else if(d >= 1) d = calc_sq_distance(x2, y2, x3, y3);
						else            d = calc_sq_distance(x2, y2, x1 + d*dx, y1 + d*dy);
					}
					if(d < m_distance_tolerance_square)
					{
						m_points.push(V2<double>(x2, y2));
						return;
					}
				}

				// Continue subdivision
				recursive_bezier(x1, y1, x12, y12, x123, y123, level + 1); 
				recursive_bezier(x123, y123, x23, y23, x3, y3, level + 1); 
			}

			void bezier(double x1, double y1, double x2, double y2, double x3, double y3)
			{
				//m_points.push(V2<double>(x1, y1)); // skip startpoint.. we use "curveTo" which implies the starting point is already there 
				recursive_bezier(x1, y1, x2, y2, x3, y3, 0);
				m_points.push(V2<double>(x3, y3));
			}

			double               m_approximation_scale;
			double               m_distance_tolerance_square;
			double               m_angle_tolerance;
			unsigned             m_count;
			tVec<V2<double> > m_points;
		};

		o3_fun void quadraticCurveTo(double cp1x, double cp1y, double x0, double y0)
		{

			V2<double> target(x0,y0);
			V2<double> cp(cp1x,cp1y);
			
			target = TransformPoint(target);
			cp = TransformPoint(cp);
			QuadraticCurveGen Gen(m_lastpoint.x,m_lastpoint.y, cp.x,cp.y, target.x, target.y);
			double x, y;

			if (m_paths.size() == 0)
			{
				m_paths.push(Path());
				m_paths[m_paths.size()-1].m_path.push(m_lastpoint);
			};

				
			while (Gen.vertex(&x,&y) != agg::agg::path_cmd_stop)
			{
				V2<double> point(x,y);
				m_paths[m_paths.size()-1].m_path.push(point);
			}

			m_lastpoint = target;

		};
		

		class BezierCurveGen
		{
		public:

			BezierCurveGen(double x1, double y1, 
					   double x2, double y2, 
					   double x3, double y3,
					   double x4, double y4) :
				m_approximation_scale(1.0),
				m_angle_tolerance(0.0),
				m_cusp_limit(0.0),
				m_count(0)
			{ 
				init(x1, y1, x2, y2, x3, y3, x4, y4);
			}

			void init(double x1, double y1, 
					  double x2, double y2, 
					  double x3, double y3,
					  double x4, double y4)
			{
				m_points.clear();
				m_distance_tolerance_square = 0.5 / m_approximation_scale;
				m_distance_tolerance_square *= m_distance_tolerance_square;
				bezier(x1, y1, x2, y2, x3, y3, x4, y4);
				m_count = 0;
			}

			unsigned vertex(double* x, double* y)
			{
				if(m_count >= m_points.size()) return agg::agg::path_cmd_stop;
				const V2<double> & p = m_points[m_count++];
				*x = p.x;
				*y = p.y;
				return (m_count == 1) ? agg::agg::path_cmd_move_to : agg::agg::path_cmd_line_to;
			}

		private:
			void recursive_bezier(double x1, double y1, 
											  double x2, double y2, 
											  double x3, double y3, 
											  double x4, double y4,
											  unsigned level)
			{
				if(level > curve_recursion_limit) 
				{
					return;
				}

				// Calculate all the mid-points of the line segments
				//----------------------
				double x12   = (x1 + x2) / 2;
				double y12   = (y1 + y2) / 2;
				double x23   = (x2 + x3) / 2;
				double y23   = (y2 + y3) / 2;
				double x34   = (x3 + x4) / 2;
				double y34   = (y3 + y4) / 2;
				double x123  = (x12 + x23) / 2;
				double y123  = (y12 + y23) / 2;
				double x234  = (x23 + x34) / 2;
				double y234  = (y23 + y34) / 2;
				double x1234 = (x123 + x234) / 2;
				double y1234 = (y123 + y234) / 2;


				// Try to approximate the full cubic curve by a single straight line
				//------------------
				double dx = x4-x1;
				double dy = y4-y1;

				double d2 = fabs(((x2 - x4) * dy - (y2 - y4) * dx));
				double d3 = fabs(((x3 - x4) * dy - (y3 - y4) * dx));
				double da1, da2, k;

				switch((int(d2 > curve_collinearity_epsilon) << 1) +
						int(d3 > curve_collinearity_epsilon))
				{
				case 0:
					// All collinear OR p1==p4
					//----------------------
					k = dx*dx + dy*dy;
					if(k == 0)
					{
						d2 = calc_sq_distance(x1, y1, x2, y2);
						d3 = calc_sq_distance(x4, y4, x3, y3);
					}
					else
					{
						k   = 1 / k;
						da1 = x2 - x1;
						da2 = y2 - y1;
						d2  = k * (da1*dx + da2*dy);
						da1 = x3 - x1;
						da2 = y3 - y1;
						d3  = k * (da1*dx + da2*dy);
						if(d2 > 0 && d2 < 1 && d3 > 0 && d3 < 1)
						{
							// Simple collinear case, 1---2---3---4
							// We can leave just two endpoints
							return;
						}
							 if(d2 <= 0) d2 = calc_sq_distance(x2, y2, x1, y1);
						else if(d2 >= 1) d2 = calc_sq_distance(x2, y2, x4, y4);
						else             d2 = calc_sq_distance(x2, y2, x1 + d2*dx, y1 + d2*dy);

							 if(d3 <= 0) d3 = calc_sq_distance(x3, y3, x1, y1);
						else if(d3 >= 1) d3 = calc_sq_distance(x3, y3, x4, y4);
						else             d3 = calc_sq_distance(x3, y3, x1 + d3*dx, y1 + d3*dy);
					}
					if(d2 > d3)
					{
						if(d2 < m_distance_tolerance_square)
						{
							m_points.push(V2<double>(x2, y2));
							return;
						}
					}
					else
					{
						if(d3 < m_distance_tolerance_square)
						{
							m_points.push(V2<double>(x3, y3));
							return;
						}
					}
					break;

				case 1:
					// p1,p2,p4 are collinear, p3 is significant
					//----------------------
					if(d3 * d3 <= m_distance_tolerance_square * (dx*dx + dy*dy))
					{
						if(m_angle_tolerance < curve_angle_tolerance_epsilon)
						{
							m_points.push(V2<double>(x23, y23));
							return;
						}

						// Angle Condition
						//----------------------
						da1 = fabs(atan2(y4 - y3, x4 - x3) - atan2(y3 - y2, x3 - x2));
						if(da1 >= pi) da1 = 2*pi - da1;

						if(da1 < m_angle_tolerance)
						{
							m_points.push(V2<double>(x2, y2));
							m_points.push(V2<double>(x3, y3));
							return;
						}

						if(m_cusp_limit != 0.0)
						{
							if(da1 > m_cusp_limit)
							{
								m_points.push(V2<double>(x3, y3));
								return;
							}
						}
					}
					break;

				case 2:
					// p1,p3,p4 are collinear, p2 is significant
					//----------------------
					if(d2 * d2 <= m_distance_tolerance_square * (dx*dx + dy*dy))
					{
						if(m_angle_tolerance < curve_angle_tolerance_epsilon)
						{
							m_points.push(V2<double>(x23, y23));
							return;
						}

						// Angle Condition
						//----------------------
						da1 = fabs(atan2(y3 - y2, x3 - x2) - atan2(y2 - y1, x2 - x1));
						if(da1 >= pi) da1 = 2*pi - da1;

						if(da1 < m_angle_tolerance)
						{
							m_points.push(V2<double>(x2, y2));
							m_points.push(V2<double>(x3, y3));
							return;
						}

						if(m_cusp_limit != 0.0)
						{
							if(da1 > m_cusp_limit)
							{
								m_points.push(V2<double>(x2, y2));
								return;
							}
						}
					}
					break;

				case 3: 
					// Regular case
					//-----------------
					if((d2 + d3)*(d2 + d3) <= m_distance_tolerance_square * (dx*dx + dy*dy))
					{
						// If the curvature doesn't exceed the distance_tolerance value
						// we tend to finish subdivisions.
						//----------------------
						if(m_angle_tolerance < curve_angle_tolerance_epsilon)
						{
							m_points.push(V2<double>(x23, y23));
							return;
						}

						// Angle & Cusp Condition
						//----------------------
						k   = atan2(y3 - y2, x3 - x2);
						da1 = fabs(k - atan2(y2 - y1, x2 - x1));
						da2 = fabs(atan2(y4 - y3, x4 - x3) - k);
						if(da1 >= pi) da1 = 2*pi - da1;
						if(da2 >= pi) da2 = 2*pi - da2;

						if(da1 + da2 < m_angle_tolerance)
						{
							// Finally we can stop the recursion
							//----------------------
							m_points.push(V2<double>(x23, y23));
							return;
						}

						if(m_cusp_limit != 0.0)
						{
							if(da1 > m_cusp_limit)
							{
								m_points.push(V2<double>(x2, y2));
								return;
							}

							if(da2 > m_cusp_limit)
							{
								m_points.push(V2<double>(x3, y3));
								return;
							}
						}
					}
					break;
				}

				// Continue subdivision
				//----------------------
				recursive_bezier(x1, y1, x12, y12, x123, y123, x1234, y1234, level + 1); 
				recursive_bezier(x1234, y1234, x234, y234, x34, y34, x4, y4, level + 1); 
			}

			//------------------------------------------------------------------------
			void bezier(double x1, double y1, 
									double x2, double y2, 
									double x3, double y3, 
									double x4, double y4)
			{
				// m_points.push(V2<double>(x1, y1)); first point skipped in "curve to"
				recursive_bezier(x1, y1, x2, y2, x3, y3, x4, y4, 0);
				m_points.push(V2<double>(x4, y4));
			}

			double               m_approximation_scale;
			double               m_distance_tolerance_square;
			double               m_angle_tolerance;
			double               m_cusp_limit;
			unsigned             m_count;
			tVec<V2<double > > m_points;
		};


		o3_fun void bezierCurveTo(double cp1x, double cp1y, double cp2x, double cp2y, double x0, double y0)
		{
			V2<double> target(x0,y0);
			V2<double> cp1(cp1x,cp1y);
			V2<double> cp2(cp2x,cp2y);
			
			target = TransformPoint(target);
			cp1 = TransformPoint(cp1);
			cp2 = TransformPoint(cp2);


			BezierCurveGen Gen(m_lastpoint.x,m_lastpoint.y, cp1.x, cp1.y, cp2.x, cp2.y, target.x, target.y);
			double x, y;

			if (m_paths.size() == 0)
			{
				m_paths.push(Path());
				m_paths[m_paths.size()-1].m_path.push(m_lastpoint);
			};


			while (Gen.vertex(&x,&y) != agg::agg::path_cmd_stop)
			{
				V2<double> point(x,y);
				m_paths[m_paths.size()-1].m_path.push(TransformPoint(point));
			}
			

			m_lastpoint = target;
		}


		o3_fun void translate(double _x, double _y)
		{
			M33<double> TransMat;
			TransMat.setTranslation(_x, _y);
			m_currentrenderstate->Transformation = m_currentrenderstate->Transformation.Multiply(TransMat);
		};

		o3_fun void rotate(double _angle)
		{
			M33<double> RotMat;
			RotMat.setRotation(_angle);//(_angle*pi*2.0f)/360.0f);
			m_currentrenderstate->Transformation = m_currentrenderstate->Transformation.Multiply(RotMat);
		};

		o3_fun void scale(double xscale, double yscale)
		{
			M33<double> ScaleMat;
			ScaleMat.setScale(xscale, yscale);
			m_currentrenderstate->Transformation = m_currentrenderstate->Transformation.Multiply(ScaleMat);
		};

		o3_fun void save()
		{
//			RenderState *PreviousState = m_currentrenderstate;
			RenderState RS = *m_currentrenderstate;
			m_renderstates.push(RS);
			m_currentrenderstate = &m_renderstates[m_renderstates.size()-1];
//			for (size_t i = 0;i<PreviousState->ClippingPaths.size();i++)
//			{
//				m_currentrenderstate->ClippingPaths.push(PreviousState->ClippingPaths[i]);
//			}
		};
		
		void RestoreStateToGraphicsObject()
		{
			m_graphics.clipBox(m_currentrenderstate->ClipTopLeft.x,
				m_currentrenderstate->ClipTopLeft.y,
				m_currentrenderstate->ClipBottomRight.x,
				m_currentrenderstate->ClipBottomRight.y);

			unsigned char *sc = (unsigned char *)&m_currentrenderstate->StrokeColor;
			m_graphics.lineColor(sc[2], sc[1], sc[0], sc[3]);
			unsigned char *fc = (unsigned char *)&m_currentrenderstate->FillColor;
			m_graphics.fillColor(fc[2], fc[1], fc[0], fc[3]);
		};
		
		o3_fun void restore()
		{
			if (m_renderstates.size()>1) 
			{
				m_renderstates.pop();
				m_currentrenderstate = &m_renderstates[m_renderstates.size()-1];
				RestoreStateToGraphicsObject();				

			};
		};

		o3_fun void setTransform(double m11, double m12, double m21, double m22, double dx, double dy)
		{
			m_currentrenderstate->Transformation.M[0][0] = m11;
			m_currentrenderstate->Transformation.M[0][1] = m12;
			m_currentrenderstate->Transformation.M[1][0] = m21;
			m_currentrenderstate->Transformation.M[1][1] = m22;
			m_currentrenderstate->Transformation.M[2][0] = dx;
			m_currentrenderstate->Transformation.M[2][1] = dy;

			m_currentrenderstate->Transformation.M[0][2] = 0;
			m_currentrenderstate->Transformation.M[1][2] = 0;
			m_currentrenderstate->Transformation.M[2][2] = 1.0;
		};

		o3_fun void transform(double m11, double m12, double m21, double m22, double dx, double dy)
		{
			M33<double> trans;
			
			trans.M[0][0] = m11;
			trans.M[0][1] = m12;
			trans.M[1][0] = m21;
			trans.M[1][1] = m22;
			trans.M[2][0] = dx;
			trans.M[2][1] = dy;

			trans.M[0][2] = 0;
			trans.M[1][2] = 0;
			trans.M[2][2] = 1.0;

			m_currentrenderstate->Transformation = m_currentrenderstate->Transformation.Multiply(trans);
		};

		o3_set void lineCap(const Str &cap)
		{
			cap;
		};

		o3_set void lineJoin(const Str &join)
		{
			join;
		};

		o3_set void miterLimit(double limit)
		{
			limit;
		};

		class ArcGen
		{
		public:
			ArcGen(double x,  double y, 
				double rx, double ry, 
				double a1, double a2, 
				bool ccw=true):
				
			m_x(x), m_y(y), m_rx(rx), m_ry(ry), m_scale(1.0)
			{
				normalize(a1, a2, ccw);
				m_path_cmd = agg::agg::path_cmd_move_to; 
				m_angle = m_start;
			}


			unsigned vertex(double* x, double* y)
			{
				if(agg::agg::is_stop(m_path_cmd)) return agg::agg::path_cmd_stop;
				if((m_angle < m_end - m_da/4) != m_ccw)
				{
					*x = m_x + cos(m_end) * m_rx;
					*y = m_y + sin(m_end) * m_ry;
					m_path_cmd = agg::agg::path_cmd_stop;
					return agg::agg::path_cmd_line_to;
				}

				*x = m_x + cos(m_angle) * m_rx;
				*y = m_y + sin(m_angle) * m_ry;

				m_angle += m_da;

				unsigned pf = m_path_cmd;
				m_path_cmd = agg::agg::path_cmd_line_to;
				return pf;
			}


		private:
			void normalize(double a1, double a2, bool ccw)
			{
				double ra = (fabs(m_rx) + fabs(m_ry)) / 2;
				m_da = acos(ra / (ra + 0.125 / m_scale)) * 2;
				if(ccw)
				{
					while(a2 < a1) a2 += pi * 2.0;
				}
				else
				{
					while(a1 < a2) a1 += pi * 2.0;
					m_da = -m_da;
				}
				m_ccw   = ccw;
				m_start = a1;
				m_end   = a2;
				m_initialized = true;
			}

			double   m_x;
			double   m_y;
			double   m_rx;
			double   m_ry;
			double   m_angle;
			double   m_start;
			double   m_end;
			double   m_scale;
			double   m_da;
			bool     m_ccw;
			bool     m_initialized;
			unsigned m_path_cmd;
		};

		V2<double> TransformPoint(V2<double> &p)
		{
			return m_currentrenderstate->Transformation.Multiply(p);
		}

		o3_fun void arc(double x0, double y0, double radius, double startAngle, double endAngle, bool anticlockwise)
		{
			
			ArcGen Gen(x0,y0,radius,radius, startAngle, endAngle, (anticlockwise)?true:false);
			double x, y;

			if (m_paths.size() == 0)
			{
				m_paths.push(Path());
			};

			if (Gen.vertex(&x,&y) != agg::agg::path_cmd_stop)
			{
				moveTo(x,y);
			};

			while (Gen.vertex(&x,&y) != agg::agg::path_cmd_stop)
			{
				V2<double> point(x,y);
				m_paths[m_paths.size()-1].m_path.push(TransformPoint(point));
			}
			
			int lastpathsize = m_paths[m_paths.size()-1].m_path.size();
			if (lastpathsize >0)
			{
				m_lastpoint = m_paths[m_paths.size()-1].m_path[lastpathsize-1];
			}
			else
			{
				m_paths[m_paths.size()-1].m_path.push(m_lastpoint);
			}

		}

		o3_fun void clip()
		{
			double x2=0,y2=0,x1=m_w,y1=m_h;
			// calculate extends, set 2d clipping rect for now
			
			
			if (m_paths.size() == 0)
			{
				m_currentrenderstate->ClippingEnabled = false;
			}
			else
			{
				m_currentrenderstate->ClippingEnabled = true;
#ifdef IMAGE_ALPHAMAP_ENABLED
				AttachAlpha();
				m_graphics.EnableAlphaMask( true ) ;

				
				agg::agg::rasterizer_scanline_aa<> m_ras;

				typedef agg::agg::renderer_base<agg::agg::pixfmt_gray8> ren_base;
				typedef agg::agg::renderer_scanline_aa_solid<ren_base> renderer;

				agg::agg::pixfmt_gray8 pixf(*m_graphics.GetAlphaBuffer());
				ren_base rb(pixf);
				renderer ren(rb);
				agg::agg::scanline_p8 m_sl;

				agg::agg::path_storage     path;

#endif

				for (size_t i = 0 ;i<m_paths.size();i++)
				{
					size_t pathlen = m_paths[i].m_path.size();
					if (pathlen >1)
					{
						{
							V2<double> &p = m_paths[i].m_path[0];
							x1 = __min(p.x, x1);
							x2 = __max(p.x, x2);
							y1 = __min(p.y, y1);
							y2 = __max(p.y, y2);
#ifdef IMAGE_ALPHAMAP_ENABLED
						path.move_to(p.x,p.y);
#endif
						}
						
						for (size_t j = 0 ;j <pathlen ;j++)
						{
							V2<double> &p = m_paths[i].m_path[j];
							x1 = __min(p.x, x1);
							x2 = __max(p.x, x2);
							y1 = __min(p.y, y1);
							y2 = __max(p.y, y2);
#ifdef IMAGE_ALPHAMAP_ENABLED
							path.line_to(p.x,p.y);
#endif
						}
#ifdef IMAGE_ALPHAMAP_ENABLED
						path.close_polygon();
#endif
							
					};
				};
#ifdef IMAGE_ALPHAMAP_ENABLED
				m_ras.add_path(path);
				ren.color(agg::agg::gray8(255));
				agg::agg::render_scanlines(m_ras, m_sl, ren);
#endif
			};
			m_currentrenderstate->ClipBottomRight.x = x2;
			m_currentrenderstate->ClipTopLeft.x = x1;
			m_currentrenderstate->ClipBottomRight.y = y2;
			m_currentrenderstate->ClipTopLeft.y = y1;
			m_graphics.clipBox(x1,y1, x2,y2);
		}
	};

	void CopyAlphaMaskToVisible()
	{

	};
};

#endif // O3_C_IMAGE1_H
