srcdir = '.'
blddir = 'build'
VERSION = '0.0.1'

def set_options(opt):
  opt.tool_options('compiler_cxx')

def configure(conf):  	
  conf.check_tool('compiler_cxx')
  conf.check_tool('node_addon')
  conf.env.append_value('CCFLAGS', ['-DDEBUG', '-g', '-O0'])
  conf.env.append_value('CXXFLAGS', ['-DDEBUG', '-g', '-O0'])

def build(bld):
  obj = bld.new_task_gen('cxx', 'shlib', 'node_addon')  
  obj.target = 'o3'
  obj.source = 'src/sh/o3_sh_node.cc'
  
  obj.includes = """
    include
    external/include    
  """
  
  obj.lib = 'xml2'