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
#ifndef O3_C_RESOURCE_BUILDER1_WIN32_H
#define O3_C_RESOURCE_BUILDER1_WIN32_H

namespace o3 {

struct cResourceBuilder1 : cScr
{
    cResourceBuilder1()
    {}

    virtual ~cResourceBuilder1()
    {}

    o3_begin_class(cScr)
    o3_end_class()

    o3_glue_gen()

    tMap<Str, siFs> m_files_to_add;    

    // new resource builder
    static o3_ext("cO3") o3_fun siScr resourceBuilder()
    {
        return o3_new(cResourceBuilder1)();
    }

    // add a file or a whole directory (recursive) to the resource builder
    o3_fun bool addAsResource(iFs* res_file,
        const char* zip_path = 0 )
    {
        if (((cFs1*) res_file)->isDir()) 
            return addDir(zip_path, res_file);

        if (((cFs1*) res_file)->isFile())
            return addFile(zip_path, res_file);

        return false;
    }

    // build the Rsc component with the files added to the RscBuilder then 
    // append the Rsc to the target file, if there was another Rsc appended
    // to the file already it will be replaced
    o3_fun void buildAndAppend(iFs* target_file, siEx* )
    {
        if (!target_file)
            return; // Ex here

        siStream stream(target_file->open("wa"));
        if (!stream)
            return; // Ex here


        Buf header, data;
        size_t length, pos(0), size_of_header, offset;
        tMap<Str, size_t> pos_of_addr;

        tMap<Str, siFs>::Iter it = m_files_to_add.begin();
        for (;it != m_files_to_add.end(); ++it) {
            Str path = (*it).key;
            length = path.size();
            
            // write path length
            header.append((void*)&length, sizeof(size_t));          

            // write path
            header.append((void*)path.ptr(), length * sizeof(char)); 

            // store pos of the data addr, and set it to 0 for now
            pos_of_addr[path] = header.size();
            header.append((void*)&pos, sizeof(size_t));

            // set size of the data to 0 for now
            header.append((void*)&pos, sizeof(size_t));
        }
        // write a trailing 0; end of the header
        header.append((void*)&pos, sizeof(size_t));

        // we will need to store the position of the zipped data
        // in the target file, that will be offset + length of the zipped
        // data before this one wheras the offset is:
        // original file size + magic num. length + header size 
        size_of_header = header.size();
        offset = ((cFs1*) target_file)->size() + sizeof(size_t) + size_of_header;

        it = m_files_to_add.begin();
        for (;it != m_files_to_add.end(); ++it) {
            Buf next_file, zipped;
            
            // read blob
            siFs fs = (*it).val;
            if (!fs) 
                return; //ex here
            next_file = ((cFs1*) fs.ptr())->blob();
            //if (ex && *ex)
            //    return;

            // zip
            ZLib::zip(next_file, zipped);            
            
            // store addr in the header
            pos = pos_of_addr[(*it).key]; 
            size_t* place_to_write = (size_t*)((uint8_t*)header.ptr() + pos);
            *place_to_write = offset + data.size();
                                        
            // store size in the header
            place_to_write++;
            *place_to_write++ = zipped.size();
            
            // append
            data.append(zipped.ptr(), zipped.size());
        }

        // set pos on the target file to point at its end
        size_t orig_size = ((cFs1*) target_file)->size();
        stream->setPos(orig_size);
        
        // write the magic num
        stream->write((void*) &cSys::rsc_magic_num, sizeof(size_t));
        
        // write header
        stream->write(header.ptr(), header.size());
        
        // write all the concatenated zipped data
        stream->write(data.ptr(), data.size());
        
        // write the pos of the magic_num
        stream->write((void*) &orig_size, sizeof(size_t));
        stream->flush();
    }

    static o3_ext("cFs1") o3_fun bool removeResource(iFs* fs_node)
    {
        siFs file(fs_node);
        if (!file || !((cFs1*) file.ptr())->isFile())
            return false;

        siStream stream = file->open("r");
        if (!stream)
            return false;

        size_t pos, magic_num, file_size;
        file_size = ((cFs1*) file.ptr())->size();
        // read the last size_t data from the end of the file
        stream->setPos(file_size - sizeof(size_t));
        if (sizeof(size_t) != stream->read((void*)&pos, sizeof(size_t)))
            return false;

        // check if its a valid position
        if (pos != stream->setPos(pos))
            return true; // no appended res, nothing to remove

        // check if it points to a rsc_magic_num
        if (sizeof(size_t) != stream->read((void*)&magic_num, sizeof(size_t))
            || cSys::rsc_magic_num != magic_num)
                return true; // no appended res, nothing to remove

        Buf original;
        original.resize(pos);
        stream->setPos(0);
        stream->read(original.ptr(), pos);
        stream->close();
        //siEx ex;
        ((cFs1*) file.ptr())->setBlob(original);
        return true; //ex == 0;
    }

    bool addDir(const char* path, iFs* dir) 
    {
        if (!((cFs1*) dir)->exists() || !((cFs1*) dir)->hasChildren())
            return false;
        
        Str new_path(path);
        if (!path || *path)            
            new_path.append(((cFs1*) dir)->name());
        
        bool success = true;

        tVec<siFs> children = ((cFs1*) dir)->children();
        for (size_t i=0; i<children.size(); i++){
            if (((cFs1*) children[i].ptr())->isFile())
                success = addFile(new_path, children[i]);
            else if (((cFs1*) children[i].ptr())->isDir())
                success = addDir(new_path, children[i]);
            
            if (!success)
                return false;
        }

        return success;
    }

    bool addFile(const char* path, iFs* file) 
    {
        Str new_path(path);
        if (!path || !*path)            
            new_path.append(((cFs1*) file)->name());

        if (!((cFs1*) file)->exists())
            return false;

        m_files_to_add[new_path] = file;
        return true;
    }
};

}

#endif // O3_C_RESOURCE_BUILDER1_WIN32_H
