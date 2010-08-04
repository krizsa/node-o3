var	component = "fs";
	o3_path = "C:/Development/o3krizsa",
	key_path = "tests/prv_key.h",
	in_path  = "build/" + component + ".dll",
	sign_path = "tests/signature",
	zip_path = "tests/" + component + ".zip",
	

// Components
	fs     	= o3.fs().get(o3_path),
	zip		= o3.zip();
// Get private key
	data = fs.get(key_path).data,

	mod     = data.match(/.*mod.*{(.*)}/)[1],
	prv_exp = data.match(/.*prv_exp.*{(.*)}/)[1];
	

function dehexify(str) {
	return str.replace(/\s|\t|\n/g, "")
		  .replace(/0x/g, "")
                  .replace(/,/g, " ")
}

mod     = o3.blob.fromHex(dehexify(mod));
prv_exp = o3.blob.fromHex(dehexify(prv_exp));

// Sign input file using private key
var dll		= fs.get(in_path),
	blob   	= dll.blob,
	digest 	= o3.sha1.hash(blob),
	signature = fs.get(sign_path),
	enc;

enc = o3.rsa.encrypt(digest, mod, prv_exp);	

// Write signed data to output file
signature.blob = enc;

zip.add(dll);
zip.add(signature);
zip.zipTo(fs.get(zip_path));