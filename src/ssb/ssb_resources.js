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

var mode = o3.args[1];

function checkFile(file) {
    if (!file.exists) {    
        o3.print("file not found: " + file.path);
        return false;
    }
    return true;
}

var src = o3.cwd.get("logo.gif");
var src2 = o3.cwd.get("logo.ico");
var src3 = o3.cwd.get("body_backg.png");
var src4 = o3.cwd.get("../../samples/o3.js");
var tgt = o3.cwd.get("../../build/ssb" + (mode == "debug" ? "_dbg.exe" : ".exe"));

checkFile(src);
checkFile(src2);
checkFile(src3);
checkFile(src4);
checkFile(tgt);

var builder = o3.resourceBuilder();
builder.addAsResource(src);
builder.addAsResource(src2);
builder.addAsResource(src3);
builder.addAsResource(src4);
builder.buildAndAppend(tgt); 