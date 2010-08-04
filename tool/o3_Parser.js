this.Lexer = {    
    scan : function(data) {
        
        var tree = [],          // parse tree
            plain_text = 0,     // block of raw text data (from c++ static string or comment)
            stack = [],         // tree stack
            type = 0,           // token type
            text_tok = 0,       // plain_text block starting token
            last_token,         // tempvar
            lines = []          // array of linepositions
            
        data.replace(TokRegexp, function(token, rx_lut, rx_ws, rx_word, rx_misc, pos) {
			// check the type
            if (rx_lut)
                type = TokenType[rx_lut];
            else if (rx_ws)
                type = TokenType.WhiteSpace;
            else if (rx_word)
                type = TokenType.Word;
            else 
                type = 0;    
            
            // if we are not inside a plain text block:
            if (!text_tok) {
                switch (type) {
                    case TokenType.Quot:
                    case TokenType.Comment:
                        tree[tree.length] = {type: type, pos: pos, token: token, plain_text: null};
                        // start a new plain text block: 
                        text_tok = token;
                        plain_text = [token];
                        break;
                        
                    case TokenType.BrackStart: 
                        tree[tree.length] = last_token = 
                            {type: type, pos: pos, token: token, subtree: null};
                        // push current tree on the stack
                        stack.push(tree);                
                        // start new tree, and the old tree (current) last element 
                        // will point to the new tree    
                        tree = last_token.subtree = [{type: 0, pos: pos, token: token}];
                        break;
                        
                    case TokenType.BrackEnd:
                        // check if the bracket match with the starting bracket
                        if (tree[0].token != TokClose[token])  {                            
                          // report error: brackets dont match
                          Reporter.error("lexer: closure token did not match, ' opened: '", 
						  tree[0].token, "' closed: '", token, "' position : ", 
							position(lines, pos).toString(),"\n");						  
                        }
                        else {
                            tree[tree.length] = {type: type, pos: pos, token: token};
                            // continue the last tree on the stack
                            tree = stack.pop();
                        }
                        break;
                        
                    case TokenType.EndOfLine: 
                        lines[lines.length] = pos;
                        break;
                        
                    case TokenType.WhiteSpace:
                        break;
                    
                    default: 
                        tree[tree.length] = {type: type, pos: pos, token: token};
                        break;
                }
            }
            // if we are in a plain_text block
            else {
                plain_text.push(token);
                switch (type) {
                    case TokenType.Quot: 
                        if (text_tok == token){
                            text_tok = 0;
                            tree[tree.length-1].plain_text = plain_text.join('');
                        }
                        break;                
                    case TokenType.Comment: 
                        if (token == '*/' && text_tok == '/*') {
                            text_tok = 0;
                            tree[tree.length-1].plain_text = plain_text.join('');
                        }
                        break;
                    case TokenType.EndOfLine: 
                        lines[lines.length] = pos;
                        if (text_tok == '//'){
                            text_tok = 0;
                            plain_text.pop();
                            tree[tree.length-1].plain_text = plain_text.join('');
                        }
                        break;
                }                
            }
        });
        while (stack.length) {
            // report error not closed
            Reporter.error("lexer: closure problem, bracket not closed.\n");
            stack.pop();
        }
        if (text_tok)
            // plain_text mode not closed      
            Reporter.error("lexer: text block not closed.\n");

        return {tree: tree, lines: lines};            
} 
}; // lexer

this.Parser = {
	readName : function(tree, i, traits) {
        Reporter.log("readName: ", tree[i].token, '\n');
         traits.___name = tree[i+1].subtree[1].plain_text;
		return 0;
    },
    readExt : function(tree, i, traits) {
        Reporter.log("readExt: ", tree[i].token, '\n');
        traits.___ext = tree[i+1].subtree[1].plain_text;
        return 0;
    },
	readProp : function(tree, i, traits) {
		var index = i,
			t;
			
		while ( (t = tree[index].token) != ';' && t != '(')
			index++;
			
		if (t == ';')
			return this.readImmProp(tree, i, traits);
		else
			return this.readFunLike(tree, i, traits);	
		
	},
    readImmProp : function(tree, i, traits){
        Reporter.log("readImmProp: ",tree[i].token,'\n');
        var name,
			index=i,
			type = tree[i].token,
			getter = (type != 'o3_set'),
			setter = (type != 'o3_get');		
		
		while ( (t = tree[index].token) != ';')
			index++;
		
		if (!(name = this.checkName(traits))) {					
			name = tree[index-1].token.replace(/^m?\_+/, "");
			name = name.replace(/_[a-z]/g, function(m,a) {
				return m.slice(1).toUpperCase();                    
			});		
		}		
			
		if (getter)
			traits.add(name , {
				imm: true,
				type:'get',
				ret: tree[i+1].token,
				toString:0,
				member: tree[index-1].token
			});
		
		if (setter)
			traits.add(name, {
				imm: true,
				type:'set',
				ret: tree[i+1].token,
				toString:0,
				member: tree[index-1].token    
			});

        return 1;
    },
    readFunLike : function(tree, index, traits) {
        Reporter.log("readFunLike: ",tree[index].token,'\n');
        var br, // position of the '(' 
            subtree,
			name,
            ret = [],
            r,i,j,l,t,tl,coma,tgt=0,
            op_count, name_pos, arg_type, arg_name, arg_def, args = [],
            eq, def;
        
        // check where the bracket starts with the args    
        for (br = index; tree[br].token != '(' && br<index+10; br++);
            
        if (br==i+5) {
            Reporter.error("tokenizer: could not find the args for the function!!!\n");
            return 1;
        }
                
        for (r = index+1; r<br-1; r++)
            ret.push(tree[r].token);
        
        // travers the function arguments
        for (subtree = tree[br].subtree, l = subtree.length, i=1; i<l; i++) {
            eq = mod = arg_def = arg_name = tgt = op_count = mod_count = 0;
            arg_type = []; 
            
            // function with no args...
            if (subtree[i].token == ')')
                break;
                
            // traverse an argument
            for (j = i; j < l-1 && subtree[j].token != ','; j++) {
                
                switch (subtree[j].token) {
                    case 'o3_tgt':
                        tgt = 1;
                        mod_count++;
                        break;
                    case 'const':
                    case 'unsigned':                    
                        mod_count++;
                        break;
                    case '*':
                    case '&':
                        op_count++;
                        break;
                    case '=': 
                        eq = j; 
                        break;                        
                }        
            }
            // at this point j is the pos of the next ',' or trailing ')'                
            
            if (eq) {
                name_pos = eq - 1;                
				if (subtree[eq+1].plain_text)
                    def = subtree[eq+1].plain_text;
                else {
                    def = subtree[eq+1].token;
					if (def == '-')
						def += subtree[eq+2].token;
					else if(def == 'Str')
						def += subtree[eq+2].token + 
							   subtree[eq+2].subtree[1].plain_text +
							   subtree[eq+3].token;
				}	
            } 
            else if (op_count + mod_count + 1 == j-i ) {
                // no name for the arg
                name_pos = null;             
            }
            else {
                name_pos = j-1;
            }
                        
            // parse the full type            
            for (t=i+tgt, tl = name_pos ? name_pos : j; t<tl; t++) {
                arg_type.push(subtree[t].token);
            }                
                        
            args.push({
                tgt : tgt,
                def : def,
                name: name_pos ? subtree[name_pos].token : null,
                type: arg_type.join(' ')
            });
            i = j;
        }
        
        var ext = this.checkExt(traits);        
		if (!(name = this.checkName(traits)))
			name = tree[br-1].token;
		
		traits.add(name, {
            type:tree[index].token.replace('o3_', ''),
            ret:ret.join(''),
            args:args,            
            ftype:0,
            ext: ext,
            name: tree[br-1].token,
			toString:0            
        });
        return 1;
    },
    checkExt : function(traits) {
        if (traits.___ext) {            
            var ret = traits.___ext;
            traits.___ext = null;
            delete traits.___ext;
            return ret.replace(/\"/g, '');
        }
        return 0;    
    },
	checkName : function(traits) {
        if (traits.___name) {            
            var ret = traits.___name;
            traits.___name = null;
            delete traits.___name;
            return ret.replace(/\"/g, '');
        }
        return 0;    
    },
    readEnum : function(tree, index, traits) {	
        var subtree = tree[index+1].subtree,i,j,l,enumName,value=-1;
        
        enumName = subtree[1].plain_text.replace(/\"/g, '');
        Enums[enumName] = 1;
        
        for (i=3; i<subtree.length; ++i) {
            for(j=i, l=subtree.length-1; j<l && subtree[j].token != ','; j++)
                ;
            
            switch(j-i) {
                case 1: // simple case
                    value++; 
                    break;
                case 3: // enum entry with value
                    value = subtree[j-1].token;
                    break
                default:
                    Reporter.error("tokenizer: syntax error in enum.\n");
            }
            
            traits.add(subtree[i].token, {
                type:'get',
                ret:'int',
                value: value,
                name: subtree[i],
                enu: enumName,
                toString:0            
            });
            
            i=j;
        }
        return 1;
    },
    
    parse : function(tree, classes, in_struct, scope) {
        var struct_found,           // struct_found found in this level
            ns_found,               // ns_found found in this level
            traits,                 // array of traits collected
            first_recur,            // call was first recursion
            tagname,                // tag name (o3_ext, o3_fun, etc.) 
            elem;                   // next tree element in the loop
                
        if (!classes) {
            classes = [];
            first_recur = 1;
        }
        
        for (var i = 0, i_len = tree.length; i < i_len; i++) {            
            elem = tree[i];
            
            if (elem.token == '{' && (ns_found || struct_found)) {
                // traverse the sub tree with a recursive call
                this.parse(elem.subtree, classes, struct_found, 
                    ns_found || struct_found);
                // class/namespace parsed, reset and continue
                ns_found = struct_found = base_class_found = null;
                continue;
            }
            
            if (elem.type != TokenType.Word) 
                continue;
                
            switch (elem.token) {
                    case 'class':
                    case 'struct':                    
                        struct_found = tree[i+1].token;        
                        break;
                    case 'namespace':
                        ns_found = tree[i+1].token;
                        break;  
                    default:
                        // skipp tokens untill we find an o3 tag
                        if (!(typename = elem.token.match(/^o3_(\w+)/)))
                            continue;
                        if (!in_struct)
                            // report error, o3_ tag out of class scope
                            continue;
                        if (!traits) {
                            // first trait for the current struct, lets create
                            // a container object for the trait of this struct
                            classes.push({
                                traits: traits = {
                                    ext : {},                   // ext traits
                                    cls : {},                   // cls traits
                                    add : function(name, o) {   // add trait function
                                        var tName = name,
                                            tType = o.ext ? 'ext' : 'cls';
                                        if (o.type == 'set')
                                            tName = name.replace(/^set(\w)/, 
                                                function(m, a){ return a.toLowerCase();});
                                          
                                        if (tName == 'toString')
                                            tName = '_toString';
                                        (this[tType][tName] ? this[tType][tName] : this[tType][tName] = []).push(o);        
                                    }   
                                },
                                struct: in_struct,
                                base:''
                            });
                        }
                        switch(typename[0]) {
                            case 'o3_name': 
                                i += this.readName(tree, i, traits);                        
                                break;
							case 'o3_prop': 
                                i += this.readImmProp(tree, i, traits);                        
                                break;
                            case 'o3_ext':
                                i += this.readExt(tree, i, traits);
                                break;        
                            case 'o3_fun':
								i += this.readFunLike(tree, i, traits);                        
                                break;
                            case 'o3_get':
                            case 'o3_set':                             
                                i += this.readProp(tree, i, traits);                        
                                break;
                            case 'o3_enum':                             
                                i += this.readEnum(tree, i, traits);                        
                                break;                                
                            case 'o3_begin_class':
                                traits.base = tree[i+1].subtree[1].token;                                    
                            case 'o3_end_class': 
                            case 'o3_add_iface':
                            case 'o3_cls':    
                                break;
                            case 'o3_glue_gen':
                                classes[classes.length-1].gen = 1;
                                // macro was not in a comment block...
                                break
                            default : 
                                // report error
                                Reporter.error("tokenizer: found unknown o3_tag : ",typename[0],"\n");
                        }
            } 
                         
                              
        }
        if (first_recur)
            return classes;
    }
}; // parser    

this.TokenType = {
    BrackStart : 2,
    BrackEnd : 3,
    Quot : 4,
    Word : 5,
    EndOfLine : 6,            
    Comment : 7,
    WhiteSpace : 8,
    '"': 4, 
    '\'': 4,
    '[': 2,
    ']': 3, 
    '{': 2, 
    '}': 3, 
    '(': 2, 
    ')': 3,
    '\n': 6, 
    '\r\n': 6, 
    '//': 7, 
    '/*': 7, 
    '*/': 7
};

this.TokClose = {'}': '{', ']': '[', ')': '('};
this.TokRegexp = /(["'{(\[\])}\]]|\r?[\n]|\/[\/*]|\*\/)|([ \t]+)|([\w._])+|(\\?[\w._?,:;!=+-\\\/^&|*"'[\]{}()%$#@~`<>])/g;
 
function posToString(){
	return 'Ln : ' + this.line + ', Col : ' + this.col;
}

function position(lines,pos) {
	for (var i = 0, j = lines.length; i < j && lines[i] < pos; i++);
	return {line: i+1, col: pos - lines[i - 1], toString: posToString};
};
	
	