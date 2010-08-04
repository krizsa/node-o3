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
#ifndef  O3_ZIP_TOOLS_H
#define O3_ZIP_TOOLS_H

#include <lib_zlib.h>


#define o3_zip_read(X,Y) if (Y != src->read((void*)X,Y))\
	return o3_new(cEx)("read from file failed.")
#define o3_zip_write(X,Y) if (Y != dest->write((void*)X,Y))\
	return o3_new(cEx)("write to file failed.")

namespace o3 {
	namespace zip_tools {
		struct Record{
			Record(const char* path, iFs* file)
				: path(path)
				, file(file)
			{

			}
			
			Str path;
			siFs file;
		};
		typedef tVec<Record> ZipRecords;
		
		static const uint32_t LH_SIGNATURE = 0x04034b50;
		static const uint32_t CH_SIGNATURE = 0x02014b50;
		static const uint32_t EH_SIGNATURE = 0x06054b50;

		struct LocalHeader{
			uint32_t signature;
			uint16_t min_version;
			uint16_t bit_flags;
			uint16_t comp_method;
			uint16_t last_mod_time;
			uint16_t last_mod_date;
			uint32_t crc32;
			uint32_t size_compressed;
			uint32_t size_uncompressed;
			uint16_t file_name_length;
			uint16_t extra_field_length;
		};

		struct CentralHeader{
			uint32_t signature;
			uint16_t version;
			uint16_t min_version;
			uint16_t bit_flag;
			uint16_t method;				
			uint16_t last_mod_time;		
			uint16_t last_mod_date;
			uint32_t crc32;
			uint32_t size_compressed;
			uint32_t size_uncompressed;
			uint16_t file_name_length;
			uint16_t extra_field_length;
			uint16_t file_comment_length;
			uint16_t disk_number;
			uint16_t internal_file_attrib;
			uint32_t external_file_attrib;
			uint32_t offset_of_file_header;
		};

		struct EndOfCentralDir{
			uint32_t signature;
			uint16_t disk_no;
			uint16_t disk_no_of_cd;
			uint16_t records_in_central_local;
			uint16_t records_in_central_total;
			uint32_t size_of_central_dir;
			uint32_t central_dir_addr;
			uint16_t comment_length;
		};

		struct CentralDir{
			tMap<Str, CentralHeader> headers;
			EndOfCentralDir end_header;
		};

		// EXAMPLE:
		//siFs node = mgr->factory("fs")(0);
		//siStream dest = node->get("dest.zip")->open("w");
		//siStream unzipped = node->get("unzipped.txt")->open("w");
		
		// ***create zip files***:
		// 1 - create a collection of files to be zipped:
		//ZipRecords records;		
		//records.push(Record("blah.txt",node->get("blah.txt")));
		//records.push(Record("blah2.txt",node->get("blah2.txt")));
		// 2 - zip the files:
		//zipMultipleFiles(records, dest);
		//dest->close();
		//siStream source = node->get("dest.zip")->open("r");

		// ***open zip files***:
		// 1 - reading the headers, from the files
		//CentralDir central_dir;
		//readCentral(source, central_dir);
		// 2 - from there on you can easily read files from it:
		//readFileFromZip(Str("blah.txt"),source,unzipped,central_dir);

// read zip file
 
		siEx findCentralDirEnd(iStream* src, EndOfCentralDir& eof_cd)
		{
			size_t size = src->size();
#ifdef O3_WIN32
			size_t max_back = min((size_t) 0xffff, size);
#else 
			size_t max_back = o3::min((size_t) 0xffff, size);
#endif
			size_t pos=(size_t)-1;
			Buf data(max_back);
			 
			src->setPos(size-max_back);
			if (max_back != src->read(data.ptr(), max_back))
				return o3_new(cEx)("reading from source failed.");
			data.resize(max_back);
			 
			pos = data.findRight(max_back-1,&EH_SIGNATURE,4);
			if (NOT_FOUND == pos)
				return o3_new(cEx)("not a valid zip file\
			- end of central directory header not found");
			 
			if (max_back-pos < 20)
				return o3_new(cEx)("not a valid zip file end of central"
					" directory header is too short");
			src->setPos(size-max_back+pos);
			o3_zip_read(&eof_cd.signature,4);
			o3_zip_read(&eof_cd.disk_no,2);
			o3_zip_read(&eof_cd.disk_no_of_cd,2);
			o3_zip_read(&eof_cd.records_in_central_local,2);
			o3_zip_read(&eof_cd.records_in_central_total,2);
			o3_zip_read(&eof_cd.size_of_central_dir,4);
			o3_zip_read(&eof_cd.central_dir_addr,4);
			o3_zip_read(&eof_cd.comment_length,2);
			if (eof_cd.disk_no != 0 || eof_cd.disk_no_of_cd != 0
				|| eof_cd.records_in_central_local !=
				eof_cd.records_in_central_total)
						return o3_new(cEx)("zip file with multiple disks not supported");
			if (eof_cd.central_dir_addr > size)
				return o3_new(cEx)("end header corrupted");
			 
			return siEx();
		}
 
		siEx parseCentralHeaders(iStream* src, CentralDir& cd)
		{
			src->setPos(cd.end_header.central_dir_addr);
			CentralHeader ch;
			size_t pos;
			for (int i=0; i < cd.end_header.records_in_central_total; i++) {
				o3_zip_read(&ch.signature, 4);
				if (ch.signature != CH_SIGNATURE)
					return o3_new(cEx)("central header not found");
				o3_zip_read(&ch.version,2);
				o3_zip_read(&ch.min_version,2);
				if (ch.min_version > 20)
					return o3_new(cEx)("zip version not supported");
				o3_zip_read(&ch.bit_flag,2);
				o3_zip_read(&ch.method,2);
				if (ch.method != 8)
					return o3_new(cEx)("zip compression method not supported");
				o3_zip_read(&ch.last_mod_time,2);
				o3_zip_read(&ch.last_mod_date,2);
				o3_zip_read(&ch.crc32,4);
				o3_zip_read(&ch.size_compressed,4);
				o3_zip_read(&ch.size_uncompressed,4);
				o3_zip_read(&ch.file_name_length,2);
				o3_zip_read(&ch.extra_field_length,2);
				o3_zip_read(&ch.file_comment_length,2);
				o3_zip_read(&ch.disk_number,2);
				o3_zip_read(&ch.internal_file_attrib,2);
				o3_zip_read(&ch.external_file_attrib,4);
				o3_zip_read(&ch.offset_of_file_header,4);
				 
				Str name(ch.file_name_length);
				o3_zip_read(name.ptr(), ch.file_name_length);
				name.resize(ch.file_name_length);
				 
				pos = src->pos() + ch.extra_field_length
				+ ch.file_comment_length;
				if (pos > src->size())
					return o3_new(cEx)("unexpected end of file");
				 
				src->setPos(pos);
				cd.headers[name] = ch;
			}
			return siEx();
		}
 
		siEx readCentral(iStream* src, CentralDir& central_dir)
		{
			siEx err;
			 
			if (err = findCentralDirEnd(src, central_dir.end_header))
				return err;
			 
			err = parseCentralHeaders(src, central_dir);
			return err;
		}
 
		//Buf readFile(iFs* source, Str path, const CentralDir& central_dir)
		//{
		//
		//}
 
		tVec<Str> listCentralDir(CentralDir& central_dir)
		{
			tVec<Str> names;
			tMap<Str, CentralHeader>::ConstIter
			it = central_dir.headers.begin(),
			end = central_dir.headers.end();
			for (;it != end; ++it) {
				names.push((*it).key);
			}
			return names;
		}
 
		siEx readFileFromZip(const Str& zip_path, iStream* src, iStream* dest, const CentralDir& central_dir)
		{
			tMap<Str,CentralHeader>::ConstIter it =
			central_dir.headers.find(zip_path);
			if (it == central_dir.headers.end() )
				return o3_new(cEx)("file not found");
			 
			const CentralHeader& ch = (*it).val;
			if (!src || !dest)
				return o3_new(cEx)("invalid file stream");
			 
			src->setPos(ch.offset_of_file_header);
			LocalHeader lh;
			o3_zip_read(&lh.signature,4);
			if (lh.signature != LH_SIGNATURE)
				return o3_new(cEx)("zip data not found.");
			 
			o3_zip_read(&lh.min_version,2);
			if (lh.min_version != ch.min_version)
				return o3_new(cEx)("zip local file header corrupted.");
			 
			o3_zip_read(&lh.bit_flags,2);
			if (lh.bit_flags & 0x01)
				return o3_new(cEx)("encrypted zip file not supported");
			// this mode could be supported... I'll migt add later support for it
			if (lh.bit_flags & 0x08)
				return o3_new(cEx)("zip mode not supported");
			if (lh.bit_flags > 8)
				return o3_new(cEx)("zip mode not supported");
			 
			o3_zip_read(&lh.comp_method,2);
			if (lh.comp_method != 8)
				return o3_new(cEx)("compression algorithm not supported (only inflate/deflate)");
			 
			o3_zip_read(&lh.last_mod_time,2);
			o3_zip_read(&lh.last_mod_date,2);
			o3_zip_read(&lh.crc32,4);
			o3_zip_read(&lh.size_compressed,4);
			o3_zip_read(&lh.size_uncompressed,4);
			o3_zip_read(&lh.file_name_length,2);
			o3_zip_read(&lh.extra_field_length,2);
			Str name(lh.file_name_length);
			o3_zip_read(name.ptr(),lh.file_name_length);
			name.resize(lh.file_name_length);
			src->setPos(src->pos()+lh.extra_field_length);
			 
			uint32_t crc;
			size_t unzipped_size = ZLib::unzip(src, dest, &crc);
			if (-1 == unzipped_size)
				return o3_new(cEx)("unzip algorithm failed.");
			if (lh.crc32 != crc)
				return o3_new(cEx)("crc32 check sum mismatch.");
			 
			return siEx();
		}
 
// write zip file
		void dosTime(iFs* node, int16_t* date, int16_t* time)
		{
#ifdef O3_WIN32			
			FILETIME last_mod, ft_local;
			ULARGE_INTEGER last_mod_ul;
			last_mod_ul.QuadPart = node->modifiedTime();
			last_mod.dwHighDateTime = last_mod_ul.HighPart;
			last_mod.dwLowDateTime = last_mod_ul.LowPart;
			FileTimeToLocalFileTime(&last_mod,&ft_local);
			FileTimeToDosDateTime(&ft_local,(LPWORD)date,(LPWORD)time);
#endif
		}
 
		siEx archiveFile(iFs* node, iStream* dest, const Str& path,
			CentralHeader& central_header )
		{
			static int16_t minversion = 20;
			static int16_t bitflag = 2;
			static int16_t method = 8;
			static int32_t zero = 0;
			int16_t date, time;
			size_t zipped_size;
			size_t size;
			siStream source;
			uint32_t crc;
			size_t name_length = path.size();
			size_t pos = dest->pos();
			size_t pos_desc,pos_back;
			 
			o3_zip_write(&LH_SIGNATURE, 4);
			o3_zip_write(&minversion, 2);
			o3_zip_write(&bitflag, 2);
			o3_zip_write(&method, 2);
			dosTime(node,&date,&time);
			o3_zip_write(&time, 2);
			o3_zip_write(&date, 2);
			pos_desc = dest->pos();
			o3_zip_write(&zero, 4); // crc
			o3_zip_write(&zero, 4); // zipped size
			o3_zip_write(&zero, 4); // size
			o3_zip_write(&name_length, 2);
			o3_zip_write(&zero, 2); // extra_field_length
			o3_zip_write(path.ptr(), name_length);
			 
			// TODO: check for big files
			size = (size_t)node->size();
			 
			if (!(source = node->open("r")))
				return o3_new(cEx)("open source file failed.");
			 
			zipped_size = ZLib::zip(source, dest,
			ZLib::Z_DEFAULT_COMPRESSION, &crc);
			 
			if (-1 == zipped_size)
				return o3_new(cEx)("zip algorithm failed");
			 
			// descriptor
			dest->flush();
			pos_back = dest->pos();
			dest->setPos(pos_desc);
			o3_zip_write(&crc, 4);
			o3_zip_write(&zipped_size, 4);
			o3_zip_write(&size, 4);
			dest->flush();
			dest->setPos(pos_back);
			 
			//central
			central_header.signature = CH_SIGNATURE;
			central_header.version = 20;
			central_header.min_version = minversion;
			central_header.bit_flag = bitflag;
			central_header.method = method;
			central_header.last_mod_time = time;
			central_header.last_mod_date = date;
			central_header.crc32 = crc;
			central_header.size_compressed = zipped_size;
			central_header.size_uncompressed = size;
			central_header.file_name_length = (uint16_t) name_length;
			central_header.extra_field_length = 0;
			central_header.file_comment_length = 0;
			central_header.disk_number = 0;
			central_header.internal_file_attrib = 0; //should be one if ASCII file...
			central_header.external_file_attrib = 0;
			central_header.offset_of_file_header = pos;
			 
			return siEx();
		}
 
		siEx writeCentralRecord(iStream* dest,
			const CentralHeader& ch, const Str& path)
		{
			o3_zip_write(&ch.signature, 4);
			o3_zip_write(&ch.version, 2);
			o3_zip_write(&ch.min_version, 2);
			o3_zip_write(&ch.bit_flag, 2);
			o3_zip_write(&ch.method, 2);
			o3_zip_write(&ch.last_mod_time, 2);
			o3_zip_write(&ch.last_mod_date ,2);
			o3_zip_write(&ch.crc32, 4);
			o3_zip_write(&ch.size_compressed, 4);
			o3_zip_write(&ch.size_uncompressed, 4);
			o3_zip_write(&ch.file_name_length, 2);
			o3_zip_write(&ch.extra_field_length, 2);
			o3_zip_write(&ch.file_comment_length, 2);
			o3_zip_write(&ch.disk_number, 2);
			o3_zip_write(&ch.internal_file_attrib, 2);
			o3_zip_write(&ch.external_file_attrib, 4);
			o3_zip_write(&ch.offset_of_file_header, 4);
			o3_zip_write(path.ptr(), path.size());
			return siEx();
		}
 
		siEx writeEndOfCentral(iStream* dest, size_t nrecords,
			size_t size, size_t offset)
		{
			static int32_t zero = 0;
			o3_zip_write(&EH_SIGNATURE,4);
			o3_zip_write(&zero,2); // num of this disk
			o3_zip_write(&zero,2); // disk of central dir
			o3_zip_write(&nrecords,2); // num of records on disk
			o3_zip_write(&nrecords,2); // num of records total
			o3_zip_write(&size,4); // size of central dir
			o3_zip_write(&offset,4); // address of central dir
			o3_zip_write(&zero,2); // comment length
			return siEx();
		}
 
		siEx zipMultipleFiles(const ZipRecords& records, iStream* dest)
		{
			//o3_assert(dest);
			siEx ret;
			tVec<CentralHeader> central_dir;
			size_t start_central, end_central;
			// write local headers and zipped data to dest
			for (size_t i=0; i<records.size(); i++) {
				central_dir.push(CentralHeader());
				const Record& rec = records[i];
				if (ret = archiveFile(rec.file, dest, rec.path,
					central_dir[central_dir.size()-1]))
						return ret;
			}
			start_central = dest->pos();
			// write central dir headers to dest
			for (size_t i=0; i<central_dir.size(); i++) {
				if (ret = writeCentralRecord(dest, central_dir[i], records[i].path))
					return ret;
			}
			end_central = dest->pos();
			// write end of central dir to dest
			ret = writeEndOfCentral(dest, central_dir.size(),
			end_central-start_central, start_central);
			 
			return ret;
		}
 
}
}
 
#undef o3_zip_write
#undef o3_zip_read

#endif // O3_ZIP_TOOLS_H