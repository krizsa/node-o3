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
#ifndef O3_C_BARCODE1_H
#define O3_C_BARCODE1_H


#include <lib_zbar.h>
namespace o3 
{

	struct cBarcode1 : cScr 
	{
		o3_begin_class(cScr)
		o3_end_class()

		o3_glue_gen()

		cBarcode1()
		{
		};

		virtual ~cBarcode1()
		{
		};

		static o3_ext("cO3") o3_get siScr barcode(iCtx* ctx)
		{
			o3_trace3 trace;
			Var blob = ctx->value("barcode");

			if (blob.type() == Var::TYPE_VOID)
				blob = ctx->setValue("barcode", o3_new(cBarcode1)());
			return blob.toScr();
		}

		o3_fun siScr __self__()
		{
			o3_trace3 trace;
			return o3_new(cBarcode1)();
		}

		static o3_ext("cImage1") o3_fun tVec<Str> scanbarcodes(o3_tgt iScr* sourceimage)
		{
			tVec<Str> Results;
			tSi<iImage> img = sourceimage;

			if (img)
			{
				
				barcode::zbar_processor_t *processor = NULL;
				processor = barcode::zbar_processor_create(0);
				if (processor == NULL) return Results;
				if(barcode::zbar_processor_init(processor, NULL, NULL)) 
				{
					barcode::zbar_processor_error_spew(processor, 0);
					return Results;
				};


				barcode::zbar_image_t *zimage = barcode::zbar_image_create();
				//assert(zimage);
				barcode::zbar_image_set_format(zimage, zbar_fourcc('Y','8','0','0'));

				int width = img->width();
				int height =img->height();
				barcode::zbar_image_set_size(zimage, width, height);

		        size_t bloblen = width * height;
		        unsigned char *blob = (unsigned char*)malloc(bloblen);
				unsigned char *d = blob;
				int startoff =0 ;
				int pixelstride = 1;
				switch (img->mode_int())
				{
				case Image::MODE_ARGB: 
					startoff = 1;
					pixelstride = 4;
					break;
				case Image::MODE_GRAY: 
					startoff = 0;
					pixelstride = 1;
					break;
				case Image::MODE_BW:
					pixelstride = 0;
					break;
				}

				// todo: make zbar take other bitmap formats as well! 
				switch (img->mode_int())
				{
				case Image::MODE_ARGB:
				case Image::MODE_GRAY:
					for (int y =0;y<height;y++)
					{
						unsigned char *row = img->getrowptr(y) + startoff;
						
						for (int x =0;x<width;x++)
						{
							*d++ = *row;
							row += pixelstride;
						}

					}
					break;
				case Image::MODE_BW:
					{
						for (int y =0;y<height;y++)
						{
							unsigned char *row = img->getrowptr(y) + startoff;
							
							for (int x =0;x<width;x++)
							{
								int shift = x&7;
								if (row[x>>3]&(1<<(7-shift)))
								{
									*d++ = 0xff;
								}
								else
								{
									*d++ = 0x00;
								};
								
							}

						}
					}
					break;
				};


				barcode::zbar_image_set_data(zimage, blob, bloblen, barcode::zbar_image_free_data);

				// copy grayscale bytes to zimage here!

				barcode::zbar_process_image(processor, zimage);
				int num_symbols =0 ;
				int found = 0;
				// output result data
				const barcode::zbar_symbol_t *sym = barcode::zbar_image_first_symbol(zimage);
		        for(; sym; sym = zbar_symbol_next(sym)) 
				{
					barcode::zbar_symbol_type_t typ = barcode::zbar_symbol_get_type(sym);
					if(typ == barcode::ZBAR_PARTIAL)
					{
						continue;
					}
					else 
					{
						Str Type(barcode::zbar_get_symbol_name(typ));
						Buf B((void*)barcode::zbar_symbol_get_data(sym), zbar_symbol_get_data_length(sym));
						Results.push(Type + ": " + Str(B));
						//printf("%s:", zbar_get_symbol_name(typ));
					};
	
				//	printf("\n");
					found++;
					num_symbols++;
				}

				barcode::zbar_image_destroy(zimage);




				barcode::zbar_processor_destroy(processor);
			};
			return Results;
		}

	};
};
#endif //O3_C_BARCODE1_H

