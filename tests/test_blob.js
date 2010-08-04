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
o3.loadModule("blob");
o3.loadModule("console");

include("test_prelude.js");

var	string = "The quick brown fox jumps over the lazy dog",
	hex    = "54 68 65 20 71 75 69 63 6B 20 62" + " " +
               "72 6F 77 6E 20 66 6F 78 20 6A 75" + " " +
               "6D 70 73 20 6F 76 65 72 20 74 68" + " " +
               "65 20 6C 61 7A 79 20 64 6F 67 00",
	base64 = "VGhlIHF1aWNrIGJyb3duIGZveCBqdW1wcyBvdmVyIHRoZSBsYXp5IGRvZwA=",

	tests = {
		fromHex : function() {
			var hexBlob = o3.blob.fromHex(hex);
			return (
					assert(hexBlob, 
						"createint blob from hex failed.")
				||	assert(hexBlob.length == (hex.length + 1) / 3, 
						"expected size of blob: " + (hex.length + 1) / 3 + " actual was : " + hexBlob.length)
				||	assert(hexBlob.toString() == string,
						"converting blob from hex back to string failed, result was: " + hexBlob.toString())
				||	assert(hexBlob.toBase64() == base64, 
						"converting blob from hex to base64 failed, result was: " + hexBlob.toBase64())
				||	true
			);
		},
		from64 : function() {
			var base64Blob = o3.blob.fromBase64(base64);
			return (
					assert(base64Blob, 
						"creating blob from base64 failed")
				||	assert(base64Blob.length == base64.length * 3 / 4 - 1,
						"expected size of blob: " + base64.length * 3 / 4 - 1 + " actual was : " + base64Blob.length)
				||	assert(base64Blob.toString() == string,
						"converting blob from base64 to string failed, result was: " + base64Blob.toString())
				||	assert(base64Blob.toHex() == hex,
						"converting blob from base64 back to base64 failed, result was: ", base64Blob.toBase64())
				|| true
			);	
		}
	}		


runScript(tests);	