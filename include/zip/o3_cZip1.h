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
#ifndef O3_CZIP1_H
#define O3_CZIP1_H

#include <lib_zlib.h>

namespace o3 {
    struct cZip1 : cScr {
		cZip1() 
		{            
        }

        o3_begin_class(cScr)
        o3_end_class();

		o3_glue_gen();

		zip_tools::ZipRecords m_records;

		static o3_ext("cO3") o3_fun siScr zip()
		{
			return o3_new(cZip1)();
		}

		// TODO: reset function
		o3_fun bool add(iFs* node, const char* zip_path = 0)
		{
			if (!node)
				return false;

			Str path = (zip_path && *zip_path) ? zip_path : node->name();

			switch(node->type()){
				case iFs::TYPE_FILE:
					m_records.push(zip_tools::Record(path,node));
					break;
				case iFs::TYPE_DIR:{					
					tVec<siFs> children = node->children();
					for (size_t i=0; i<children.size(); i++){
						Str newpath = path;
						newpath.append("/");
						newpath.append(children[i]->name());
						if (!add(children[i], newpath))
							return false;
					}
					break;
				}					
				default:
					return false;
			}
			return true;
		}

		o3_fun bool zipTo(iFs* dest, siEx* ex=0)
		{ 
			if (!dest)
				return false;

			siStream dest_stream = dest->open("w");
			if (!dest_stream)
				return false;

			siEx error = zip_tools::zipMultipleFiles(m_records, dest_stream);
			if (ex) 
				*ex = error;			

			return !error;
		}
	};

	struct cUnzip1 : cScr {

		cUnzip1() 
		{            
		}

		o3_begin_class(cScr)
		o3_end_class();

		o3_glue_gen();

		zip_tools::CentralDir m_central_dir;
		siStream m_stream;

		static o3_ext("cO3") o3_fun siScr unzip()
		{
			return o3_new(cUnzip1)();
		}

		o3_fun bool openZipFile(iFs* zip_file, siEx* ex=0)
		{
			if (!zip_file)
				return false;

			return openZipFromStream(zip_file->open("r"), ex);
		}

		o3_fun bool openZipFromStream(iStream* zip_stream, siEx* ex=0) 
		{
			if (!zip_stream)
				return false;

			m_stream = zip_stream;
			siEx error = zip_tools::readCentral(
				m_stream, m_central_dir);
			
			if(error){
				if(ex)
					*ex = error;
				m_central_dir = zip_tools::CentralDir();
				return false;
			}
			return true;
		}

		o3_get tVec<Str> listFiles()
		{
			return zip_tools::listCentralDir(m_central_dir);
		}

		Buf get(const Str& zip_path, siEx* ex=0)
		{
			cBufStream* buf_stream = o3_new(cBufStream)();
			get(zip_path, siStream(buf_stream), ex);
			return buf_stream->buf();
		}

		o3_fun bool get(const Str& zip_path, iStream* dest, siEx* ex=0)
		{
			if (!m_stream || !dest)
				return false;

			siEx error = zip_tools::readFileFromZip(
				zip_path, m_stream, dest, m_central_dir);
			
			if (ex)
				*ex = error;
			return !error;
		}

		o3_fun void unzip(iFs* from, iFs* to)
		{
			from;to;
			// TODO: implement
		}
	};
}

#endif // O3_CZIP1_H
