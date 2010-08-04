#!/bin/o3
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
 
var exit = false; 
o3.loadModule("console");
o3.loadModule("http");

url = "http://en.wikipedia.org/wiki/"
    + o3.args[1].charAt(0).toUpperCase()
    + o3.args[1].substr(1);

o3.print(url);    
    
http = o3.http();
http.open("GET", url);
http.onprogress = function() {
    o3.print(".");
}
http.onreadystatechange = function() {
    if (http.readyState == http.READY_STATE_COMPLETED) {
        o3.print("done\n");
        http.responseText.replace(/<p>(.*)<\/p>/, function(str) {
            o3.print(str.replace(/<.*?>/g, "") + "\n");
        });
        exit = true;
    }
}
http.send("");
while(!exit)
    o3.wait(10);
