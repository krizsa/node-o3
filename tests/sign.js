// Configuration settings

var	key_path = "C:/Development/o3krizsa/tests/prv_key.h",
	in_path  = "C:/Development/o3krizsa/build/fs.dll",
	out_path = "C:/Development/o3krizsa/tests/signature",

// Components
	fs     = o3.fs(),

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
var blob   = fs.get(in_path).blob,
	digest = o3.sha1.hash(blob),
	enc;

enc = o3.rsa.encrypt(digest, mod, prv_exp);	

// Write signed data to output file
fs.get(out_path).blob = enc;