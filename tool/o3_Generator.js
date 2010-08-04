this.Generator = {
    //properties:
    //{
    //      cls: {
    //          prop1: [trait1,trait2,...],
    //          prop2: [...]
    //          ...
    //      }
    //      ext: {
    //          prop1: [...],
    //          ...
    //      }
    // }
 
    // Generates the whole glue file for a class
    run : function(className, properties) {
        var t=[];
        t.push('#ifdef O3_WITH_GLUE\n',
            '#pragma warning( disable : 4100)\n',
            '#pragma warning( disable : 4189)\n',            
            'namespace o3 {\n\n\n',        
            this.genSelect(className),
            this.genTables(className, properties),
            this.genInvokes(className, properties),
            '}\n#endif\n',
            '#pragma warning(default : 4100)\n',
            '#pragma warning(default : 4189)\n');
        return t.join('');    
    },
    
    // Generates the select function
    genSelect : function(className) {
        var t = [];
        t.push('Trait* ',className,'::select()\n{\n   return clsTraits();\n}\n\n');
        return t.join('');
    },

    // Generates the tables
    genTables : function(className, properties) {
        function wss(n) {
            var i,t=[];
            for (i=0; i<n; i++)
                t.push(' ');
            return t.join('');            
        }

        var uid=0,  // unique id (for properties) (Trait::offset)
        sid=0,      // serial id (for traits)     (Trait::index) 
        ext=0,      
        set,        // set of properties to generater invoke for 
        prop,propName, // current property in the loop and its name
        i,i_l,  
        tr,         // current trait for the current property in the loop
        tType,      // first round its 'cls' then 'ext' in the second
        clsName,    // name of the class for cls traits, name of the class
                    // to extend for ext traits
        t=[];       // table
                   
        // traverse first the ext then the cls properties
        for (;ext<2;ext++) {                       
            tType = ext?'ext':'cls';
            set = properties[tType];
            
            t.push('Trait* ', className,'::', tType , 'Traits()\n',
                '{\n', '      ', 'static Trait TRAITS[] = {\n');
            t.push('         {      0,      Trait::TYPE_BEGIN,      "', className, 
                '",',wss(19-className.length),'0,',wss(20),'0,              0,      ', 
                (ext ? '0                  }' : (properties.base + '::clsTraits()  }')),
                ',\n');    
            // traverse the selected set of properties by trait names
            for (propName in set) {
                prop = set[propName];
                prop.uid = uid;
                // bahhh temp. workaround for a stupid IE bug...
                if (propName == '_toString')
                    propName = 'toString';
                
                // traverse the set of properties that has the same name 
                for (i=0, i_l=prop.length; i<i_l; i++) {
                        tr = prop[i];
                        tr.sid = sid++;        
                        clsName = ext ? tr.ext : className;
                        t.push('         {      ', uid, uid > 9 ? ',     ' : ',      ',
                            'Trait::TYPE_', tr.type.toUpperCase(), ',        "', 
                            clsName, '",',wss(19-clsName.length),'"', propName, 
                            '",', wss(19-propName.length), tType, 'Invoke,      ', 
                            tr.sid, tr.sid > 9 ? ',     ' : ',      ','0                  },\n' );
                }
                uid++;
            }
            t.push('         {      0,      Trait::TYPE_END,        "', className, 
                '",',wss(19-className.length),'0,',wss(20),'0,              0,      ',
                '0                  }', ',\n');
            t.push('      };\n\n','      ','return TRAITS;\n}\n\n');                        
            uid = sid = 0;        
        }
        
        return t.join('');
    },
    
    // Generates the invoke functions
    genInvokes : function(className, properties) {
        var t = [],
            tType,
            set,
            propName,
            prop,
            hasProps = {ext:false, cls:false},
            ext;
        
        // check if there are any properties at all...
        for (ext=0;ext<2;ext++) { 
            tType = ext?'ext':'cls';
            for (var i in properties[tType]) {
                hasProps[tType] = true;
                break;
            }
        }
        
        // generate the clsInvoke and extInvoke functions    
        for (ext=0;ext<2;ext++) { 
            tType = ext?'ext':'cls';
            set = properties[tType];            
            t.push('siEx ',className,'::',tType,'Invoke(iScr* pthis, iCtx* ctx, int index, int argc,\n',
                   '           const Var* argv, Var* rval)\n{\n');
            // against unreference arguments warning: 
            t.push('      siEx ex;\n');
            // generate the switch only if there are any properties
            if (hasProps[tType]) {
                t.push('      ', className, '* pthis1 = (', className, '*) pthis;\n\n');
                t.push('      ', 'switch(index) {\n');
                for (propName in set) {
                    prop = set[propName];
                    this.genBlockForProp(prop,t,'         ');        
                }    
                t.push('      }\n');
            }
            t.push('      ','return ex;\n}\n\n');
        }
        
        return t.join('');    
    },
    
    // Generates all the 'case' blocks in the invoke function for a property
    genBlockForProp : function(prop,t,ws) {                
        function overloadType(prop) {
            if (prop.length < 2)
                return 0;
            if (prop.length == 2) { 
                var t1 = prop[0].type,
                    t2 = prop[1].type;
                if (  (t1 = 'get' &&  t2 == 'set') 
                    ||(t1 = 'set' &&  t2 == 'get') )
                        return 0;
                return 1;        
            }            
            for (var i in prop)
                if (prop[i].type == 'get') 
                    return 2;
            return 1;                
        };
    
        var v, trait, first, overloads = [], ws2 = ws + '   ';
        
        // for one property there can belong more traits, 
        // a getter and a setter for example, or several overloads
        // for the same function, or a getter and several overloads
        // for the setter function...
        switch(overloadType(prop)){
            // no overloads
            case 0:
                for (v=0;v<prop.length; v++) {
                    trait = prop[v];
                    t.push(ws, 'case ', trait.sid, ':\n');
                    this.genBlockForTrait(trait,t,ws2, true);
                    t.push('\n',ws2, 'break;\n');
                }                    
                break;
            
            // overloaded function / setter
            case 1:                
                t.push(ws, 'case ', prop[0].sid, ':\n');
                this.genBlockForOverloads(prop, t, ws2);
                t.push(ws2, 'break;\n');
                break;
            
            // overloaded setter with a getter
            case 2:
                for (v=0;v<prop.length; v++) {
                    trait = prop[v];
                    if (trait.type == 'get') {
                        t.push(ws, 'case ', trait.sid, ':\n');
                        this.genBlockForTrait(trait,t,ws2);
                        t.push('\n',ws2, 'break;\n');
                    }
                    else { 
                      overloads.push(trait);  
                      if (!first)
                        first = v;    
                    }
                }
                t.push(ws, 'case ', prop[first].sid, ':\n');
                this.genBlockForOverloads(overloads, t, ws2);
                t.push('\n',ws2, 'break;\n');
                break;
            
            // error
            default:
                // TODO: implement checking for invalid overloads...
                Reporter.error('generator: illegal overload');
        }
    },
    
    
    genBlockForTrait : function(trait,t,ws,check) {        
        function genImmCall(trait) {
            if (trait.type == 'get')
                return {call: 'pthis1->' + trait.member};
            
            return genArgsForCall({args: [{type:trait.ret}]});     
        };
        function genEnumCall(trait) {            
            return trait.value;
        };

        // '*rval = ' type of assignement
        function genCallWrapper(trait) {
            var siSg, base;
            
            if (trait.imm && trait.type == 'set') {
                return {start:'pthis1->' + trait.member + ' = ', close:';'};
            }
            
            switch (trait.ret) {
                case 'void': 
                    return {start:'', close:';'};
                case 'bool':    case 'int':     case 'int32_t':
                case 'int64_t': case 'size_t':  case 'double':
                case 'Str':     case 'WStr':    case 'siScr':
                case 'Var':
                    return {start:'*rval = ', close:';'};
                case 'Buf':
                    return {start:'*rval = o3_new(cScrBuf)(', close:');'};            
                default: {
                    if (trait.ret.indexOf('tVec') != -1) {
                        base = trait.ret.match(/<[\w]+>/);
                        base = base[0].substring(1,base[0].length-1);
                        return {start:'*rval = o3_new(tScrVec<' + base + '>)(', close:');'};
                    }                    
                    if (siSg = trait.ret.match(/si[\w]+/)){
                        return {start:'*rval = ' + siSg[0] + '(', close:');'};
                    }
                    if (Enums[trait.ret]) {
                        return {start:'*rval = ', close:';'};
                    }
                    Reporter.error('generator: unknown return type: ', trait.ret, '\n');
                }
            
            }
        };
        // arguments for the function call + arg count check    
        function genArgsForCall(trait) {
            var args = trait.args, i, min=-1, max=0, fetch='', spec_arg=false, def_start='', def_close='', 
            wrap_start='', wrap_close='', argc_check=[], call = [],info,arglist=false;    
            
            for (i=0; i<args.length; i++) {
                if (!(info = ArgInfo[args[i].type]))
                    info = ArgInfo.si(args[i].type)
                 
				if (info.arglist) {
					arglist=true;
					call.push(info.fetch);					
					call.push(',');
					if (!args[i+1] || args[i+1].type != 'int')
						Reporter.error('generator: genArgsForCall failed: Var* as '
							+'function argument must be followed by an int argument (argc)');
					i++; 
					continue;
				}
				 
                fetch = args[i].tgt ? 'pthis' : info.fetch;
                wrap_start = info.wrap ? info.wrap + '(' : '';
                wrap_close = info.wrap ? ')' : '';

				// second wrapper for the Buf...
                if (info.wrap2) {
                    wrap_start = info.wrap2 + '(' + wrap_start;
                    wrap_close += ')';
                }     

				// not real script argument like siEx*, iCtx*, o3_tgt, etc.
                spec_arg = info.type ? (args[i].tgt ? true : false) : true;
                
                if (args[i].def && !spec_arg) {
                    if (min<0)
                        min = max;
                    // if it was an siEx* param for example we dont want to check the arg count
					def_start = spec_arg ? '' : ('argc > ' + max + ' ? ');
                    def_close = ' : ' + args[i].def;
                }
                    
                call.push(wrap_start, def_start);
                if (!spec_arg) {
                    call.push('argv[',max++,']');
					if (!info.direct)	
						call.push('.');
				}	
				
                call.push(fetch,def_close,wrap_close, ',');            
                                    
                iSg=fetch=def_start=def_close=wrap_start=wrap_close='';
                spec_arg= false;        
            }
            if (args.length > 0)
                call.pop(); // remove last ',' 
            			
            if (min>0)
                argc_check.push('argc < ', min, ' && ');
            if (min==-1)
                argc_check.push('argc != ', max);
            else
                argc_check.push('argc > ', max );
            
            return {
					call: call.join(''), 
					argc_check: arglist ? null : argc_check.join('')
				};
					
            // like: {call: 'ctx, argv[0].toInt(), &ex', argc_check: 'argc!=1'}
        };    
        
        var args, wrapper = genCallWrapper(trait);
        
        if (trait.imm)
            t.push(ws, wrapper.start, genImmCall(trait).call, wrapper.close)
        else if (trait.enu)
            t.push(ws, wrapper.start, genEnumCall(trait), wrapper.close)
        else { 
            args = genArgsForCall(trait);
            if (check)             
                t.push(ws, 'if (', args.argc_check, ')\n', ws, '   ', 
                    'return o3_new(cEx)("Invalid argument count. ( ',trait.name,' )");\n');
            
            t.push(ws, wrapper.start, 'pthis1->',
                trait.cName ? trait.cName : trait.name, '(', args.call, ')' , wrapper.close);
        }
    },
    
    genBlockForOverloads : function(overloads, t, ws) {
        function classifyArgs(fun) {            
            var min=0, max=0, spec=[], scr=[], foundDef, arg, varInfo;
            for (var i=0; i<fun.args.length; i++) {
                arg = fun.args[i];
                if (arg.tgt) {
                    spec.push(arg);
                    continue;
                }                
                
                varInfo = ArgInfo[arg.type] ? ArgInfo[arg.type] : ArgInfo.si(arg.type);
                                                
                if ( varInfo && varInfo.type ) {
                    arg.si=varInfo.wrap;
                    arg.varType = varInfo.type;
                    scr.push(arg);
                    if (arg.def)
                        foundDef = true;                    
                    max++;
                    if (!foundDef)
                        min++;
                }                
                
                else {
                    spec.push(arg);
                    continue;
                }                                            
            }
            fun.specArgs = spec;    // helper arguments of the c++ function (ex,ctx,pthis)
            fun.scrArgs = scr;      // arguments of the js method
            fun.min = min;          // min argc for the js method
            fun.max = max;          // max argc for the js method
        };
        // compares two array of overloads
        function funListEqual(funs1, funs2) {
            diff = false;
            if (funs1.length == funs2.length) {
                for (var j=0; j<funs1.length; j++) 
                    if (funs1[j] != funs2[j]) {
                        diff = true;
                        break;        
                    }
			}		
			else 
				diff = true;		
            return !diff;        
        };
        // grouping overloads based on their possible argument counts
        // example: with 1 arg overload1 and overload2 can be called,
        //          with 2-3 args only overload2 can be called, etc...
        function argcPartition(funs) {
            var valid = {}, ret = [],
                group;
            
            for (var f=0; f<funs.length; f++)
                for (var i=funs[f].min; i<funs[f].max+1; i++)
                    (valid[i] ? valid[i] : valid[i]=[]).push(funs[f]);                                       
                
            for (var i in valid) {
                if (group && funListEqual(group, valid[i])) {
                    ret[ret.length-1].max = i;                        
                    continue;
                }
                
                else {                
                    group = valid[i];
                    ret.push({
                        fun : group,
                        min : i,
                        max : i
                    });
                    continue;    
                }                                                    
            }
            
            return ret;
        };
        // grouping overloads based on the type of the 'index'-th script argument
        // example: index = 2
        //          if the type (=t) of 2nd script argument is 
        //          TYPE_NULL <= t >=TYPE_INT32 onverload1 must be called 
        //          else either overload2 or overload3 can be called check the 3rd arg
        function typePartition(funs, index) {
            var i,blocks=[],type,sorted = {si:{}},isSi,si,last=-1;
            
            for (i=0; i<funs.length; i++) {
                type = funs[i].scrArgs[index].varType;
                si = funs[i].scrArgs[index].si;
                
                if (type && !si)
                    (sorted[type] ? sorted[type] : sorted[type] = []).push(funs[i]);
                else if (type){
                    (sorted.si[si] ? sorted.si[si] : sorted.si[si] = []).push(funs[i]);
                } else {
                    Reporter.error('generator: type partition failed: ',type,'\n');
                }
            }                                                
                            
            for (i in Types) {
                type = Types[i];                            
                
                if (!sorted[type])
                    continue;
                
                if (type == 'STR') {
                    // stupid exception case, dont know how to get rid of it yet...
                    ++i;
                } 
                
                blocks.push({
                    fun : sorted[type],
                    min : Types[last+1], 
                    max : Types[i]
                });                               
                
                last = i*1;                                                
            }

            if (blocks.length)
                blocks[blocks.length-1].max = Types[Types.length-1];              
            
            for (si in sorted.si) {
                blocks.push({
                    fun : sorted.si[si],
                    si : si  
                });
                continue;
            }
            
            return blocks;
        };
        function recursiveGen(blocks, ws, t, level) {
            var i,j,si,siVar,funs,b,toCheck = 'argc', min, max, index = 0;
            with(Generator) {                        
                if (level) {
                    toCheck = 'type' + (level-1);
                    t.push(ws, 'Var::Type ',toCheck,' = argv[',level-1,
                        '].type();\n');

                    for (i=0; i<blocks.length; i++) {                        
                        if (si = blocks[i].si) {
                            siVar = si.toLowerCase(),'_', level-1;                            
                            
                            t.push(ws, index++ ? 'else if(' : 'if (' ,si, ' ', siVar,
                                ' = ', si, '(argv[', level-1, '].toScr())',') {\n');                                                            
                        
                            funs = blocks[i].fun;
                            if (funs.length == 1) {
                                genBlockForTrait(funs[0],t,ws + '   ');
                                t.push('\n');
                            } 
                            else {
                                recursiveGen(typePartition(funs, level),
                                    ws + '   ', t, level+1);
                            } 
                            t.push(ws, '   return ex;\n')
                            t.push(ws, '}\n');
                        }    
                    }
                } 
                
                for (i=0; i<blocks.length; i++) {
                    b = blocks[i];
                    if (b.si)
                        continue; // already handled
                    
                    min = level ? "Var::TYPE_" + b.min : b.min;
                    max = level ? "Var::TYPE_" + b.max : b.max;
                    t.push(ws, index++ ? 'else if(' : 'if (');
                    if (b.min == b.max) {
                        t.push(toCheck, '==', min, ') {\n');
                    }
                    else {
                        t.push(min, ' <= ', toCheck, 
                            ' && ' , max,' >= ', toCheck, ') {\n');                        
                    }
                    
                    if (b.fun.length == 1) {
                        genBlockForTrait(b.fun[0],t,ws + '   ');
                        t.push('\n');
                    }
                    else {
                        recursiveGen(typePartition(b.fun, level),
                                ws + '   ', t, level+1);
                    }
                    
                    t.push(ws,'}\n');                        
                }
				
				if (blocks.length)
					t.push(ws,'else{\n',ws,'   return o3_new(cEx)(', 
						level ? '"Invalid argument type."' : '"Invalid argument count."',
						');\n', ws,  '}\n');
            }    
        };
        
        for (var i=0; i<overloads.length; i++)    
            classifyArgs(overloads[i]);
            
        return recursiveGen(argcPartition(overloads), ws, t, 0);                
    }
}; // generator

this.ArgInfo = {
    'bool'              : {fetch:'toBool()',    type:'BOOL'},
    'int'               : {fetch:'toInt32()',   type:'INT32'},
    'int32_t'           : {fetch:'toInt32()',   type:'INT32'},
    'size_t'            : {fetch:'toInt32()',   type:'INT32'},
    'int64_t'           : {fetch:'toInt64()',   type:'INT64'},
    'double'            : {fetch:'toDouble()',  type:'DOUBLE'},
    'const char *'       : {fetch:'toStr()',     type:'STR'},
    'const Str &'       : {fetch:'toStr()',     type:'STR'},
    'const wchar_t *'   : {fetch:'toWStr()',    type:'WSTR'},
	'const Var &'		: {fetch:'', 			type:'VAR', 
		direct: true},
    'Var *'				: {fetch:'argv,argc',	arglist: true},
	'const WStr &'      : {fetch:'toWStr()',    type:'WSTR'},
    'const Buf &'       : {fetch:'toScr()',     type:'SCR', 
        wrap : 'siBuf', wrap2 : 'Buf'},
    'iScr *'            : {fetch:'toScr()',     type:'SCR'},
    'iCtx *'            : {fetch:'ctx'},
    'siEx *'            : {fetch:'&ex'},

    si : function(si) {
        if (si.match(/i[\w]+ \*/)) 
            return {
                fetch : 'toScr()',
                wrap : 's' + si.substring(0,si.indexOf('*')),
                type : 'SCR'
            }
        else
            Reporter.error('generator: unknown arg type: ',si,'\n');
        return {};    
    }    
};

this.Types = [
      'VOID',
      'NULL',
      'BOOL',
      'INT32',
      'INT64',
      'DOUBLE',
      'STR',
      'WSTR',
      'SCR'           
];   

this.Enums = {};
