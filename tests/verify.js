// Configuration settings
key_path = "C:\\Development\\o3modules\\tests\\pub_key.h";
in_path  = "C:\\Development\\o3modules\\tests\\license.out";

// Components
var fs     = o3.fs();

// Get public key
var data = fs.get(key_path).data;

var mod     = data.match(/.*mod.*{(.*)}/)[1];
var pub_exp = data.match(/.*pub_exp.*{(.*)}/)[1];

function dehexify(str) {
	return str.replace(/\s|\t|\n/g, "")
		  .replace(/0x/g, "")
                  .replace(/,/g, " ")
}

mod     = o3.blob.fromHex(dehexify(mod));
pub_exp = o3.blob.fromHex(dehexify(pub_exp));

// Verify input file using public key
var data = fs.get(in_path).data;

var text = data.match(/.*(?=\n\n)/)[0];
var digest = data.substring(data.indexOf('\n\n')+2); //data.match(/\n\n(.*)/)[1];

var datablob = o3.blob.fromBase64(digest);
digest = o3.rsa.decrypt(datablob, mod, pub_exp);

if (o3.sha1.hash(text).toHex() == digest.toHex())
	o3.print("File is valid");
else
	o3.print("File is not valid");