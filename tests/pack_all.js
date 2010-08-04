var	component = "fs";
	o3_path = "C:/Development/o3krizsa",
	key_path = "tests/prv_key.h",
	version = "v0_9",
// Components
	fs     	= o3.fs().get(o3_path),
	hash_zip		= o3.zip();
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

function zipAndSign(dll, signature, hash, output, hash_zip) {
	var zip	= o3.zip();
	blob = dll.blob,
	digest 	= o3.sha1.hash(blob),
	enc = o3.rsa.encrypt(digest, mod, prv_exp);	

	hash.blob = digest;
	signature.blob = enc;
	
	zip.add(dll, dll.name.replace(".dll", version + ".dll"));
	zip.add(signature);
	zip.zipTo(output);
	
	hash_zip.add(hash);
}

var sig = fs.get("tests/sign"), 
out = fs.get("tests/out"),
hash = fs.get("tests/hash"),
bin = fs.get("build"),
children = bin.children,
f, name;

o3.print(sig.path + "\n" + out.path + "\n" + bin.path + "\n" + children.length + "\n");

for (var i=0; i<children.length; i++) {
	f = children[i];
	name = f.name;
	if (name.indexOf(".dll")!=-1 && name.indexOf("npplugin")==-1 ) {				
		zipAndSign(f,sig.get(name.replace(".dll", version)),
			hash.get(name.replace(".dll", version)),
			out.get(name.replace(".dll", version + ".zip")), hash_zip);
	}
	hash_zip.zipTo(out.get("hash.zip"));	
}
