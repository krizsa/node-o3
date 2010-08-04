#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cFs1Base::select()
{
   return clsTraits();
}

Trait* cFs1Base::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cFs1Base",           0,                    0,              0,      cScr::clsTraits()  },
         {      0,      Trait::TYPE_GET,        "cFs1Base",           "INVALID",            clsInvoke,      0,      0                  },
         {      1,      Trait::TYPE_GET,        "cFs1Base",           "DIR",                clsInvoke,      1,      0                  },
         {      2,      Trait::TYPE_GET,        "cFs1Base",           "FILE",               clsInvoke,      2,      0                  },
         {      3,      Trait::TYPE_GET,        "cFs1Base",           "LINK",               clsInvoke,      3,      0                  },
         {      4,      Trait::TYPE_GET,        "cFs1Base",           "valid",              clsInvoke,      4,      0                  },
         {      5,      Trait::TYPE_GET,        "cFs1Base",           "exists",             clsInvoke,      5,      0                  },
         {      6,      Trait::TYPE_GET,        "cFs1Base",           "type",               clsInvoke,      6,      0                  },
         {      7,      Trait::TYPE_GET,        "cFs1Base",           "isDir",              clsInvoke,      7,      0                  },
         {      8,      Trait::TYPE_GET,        "cFs1Base",           "isFile",             clsInvoke,      8,      0                  },
         {      9,      Trait::TYPE_GET,        "cFs1Base",           "isLink",             clsInvoke,      9,      0                  },
         {      10,     Trait::TYPE_GET,        "cFs1Base",           "accessedTime",       clsInvoke,      10,     0                  },
         {      11,     Trait::TYPE_GET,        "cFs1Base",           "modifiedTime",       clsInvoke,      11,     0                  },
         {      12,     Trait::TYPE_GET,        "cFs1Base",           "createdTime",        clsInvoke,      12,     0                  },
         {      13,     Trait::TYPE_GET,        "cFs1Base",           "size",               clsInvoke,      13,     0                  },
         {      14,     Trait::TYPE_GET,        "cFs1Base",           "path",               clsInvoke,      14,     0                  },
         {      15,     Trait::TYPE_GET,        "cFs1Base",           "name",               clsInvoke,      15,     0                  },
         {      15,     Trait::TYPE_SET,        "cFs1Base",           "name",               clsInvoke,      16,     0                  },
         {      16,     Trait::TYPE_FUN,        "cFs1Base",           "get",                clsInvoke,      17,     0                  },
         {      17,     Trait::TYPE_GET,        "cFs1Base",           "parent",             clsInvoke,      18,     0                  },
         {      18,     Trait::TYPE_GET,        "cFs1Base",           "hasChildren",        clsInvoke,      19,     0                  },
         {      19,     Trait::TYPE_FUN,        "cFs1Base",           "scandir",            clsInvoke,      20,     0                  },
         {      20,     Trait::TYPE_GET,        "cFs1Base",           "children",           clsInvoke,      21,     0                  },
         {      21,     Trait::TYPE_FUN,        "cFs1Base",           "createDir",          clsInvoke,      22,     0                  },
         {      22,     Trait::TYPE_FUN,        "cFs1Base",           "createFile",         clsInvoke,      23,     0                  },
         {      23,     Trait::TYPE_FUN,        "cFs1Base",           "createLink",         clsInvoke,      24,     0                  },
         {      24,     Trait::TYPE_FUN,        "cFs1Base",           "remove",             clsInvoke,      25,     0                  },
         {      25,     Trait::TYPE_FUN,        "cFs1Base",           "copy",               clsInvoke,      26,     0                  },
         {      26,     Trait::TYPE_FUN,        "cFs1Base",           "move",               clsInvoke,      27,     0                  },
         {      27,     Trait::TYPE_FUN,        "cFs1Base",           "open",               clsInvoke,      28,     0                  },
         {      28,     Trait::TYPE_GET,        "cFs1Base",           "canRead",            clsInvoke,      29,     0                  },
         {      29,     Trait::TYPE_GET,        "cFs1Base",           "canWrite",           clsInvoke,      30,     0                  },
         {      30,     Trait::TYPE_GET,        "cFs1Base",           "blob",               clsInvoke,      31,     0                  },
         {      30,     Trait::TYPE_SET,        "cFs1Base",           "blob",               clsInvoke,      32,     0                  },
         {      30,     Trait::TYPE_SET,        "cFs1Base",           "blob",               clsInvoke,      33,     0                  },
         {      31,     Trait::TYPE_GET,        "cFs1Base",           "data",               clsInvoke,      34,     0                  },
         {      31,     Trait::TYPE_SET,        "cFs1Base",           "data",               clsInvoke,      35,     0                  },
         {      32,     Trait::TYPE_GET,        "cFs1Base",           "onchange",           clsInvoke,      36,     0                  },
         {      32,     Trait::TYPE_SET,        "cFs1Base",           "onchange",           clsInvoke,      37,     0                  },
         {      33,     Trait::TYPE_GET,        "cFs1Base",           "fopen",              clsInvoke,      38,     0                  },
         {      34,     Trait::TYPE_GET,        "cFs1Base",           "fseek",              clsInvoke,      39,     0                  },
         {      35,     Trait::TYPE_GET,        "cFs1Base",           "fread",              clsInvoke,      40,     0                  },
         {      36,     Trait::TYPE_GET,        "cFs1Base",           "fwrite",             clsInvoke,      41,     0                  },
         {      37,     Trait::TYPE_GET,        "cFs1Base",           "fflush",             clsInvoke,      42,     0                  },
         {      38,     Trait::TYPE_GET,        "cFs1Base",           "fclose",             clsInvoke,      43,     0                  },
         {      0,      Trait::TYPE_END,        "cFs1Base",           0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cFs1Base::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cFs1Base",           0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_END,        "cFs1Base",           0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cFs1Base::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cFs1Base* pthis1 = (cFs1Base*) pthis;

      switch(index) {
         case 0:
            *rval = 0;
            break;
         case 1:
            *rval = 1;
            break;
         case 2:
            *rval = 2;
            break;
         case 3:
            *rval = 3;
            break;
         case 4:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( valid )");
            *rval = pthis1->valid();
            break;
         case 5:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( exists )");
            *rval = pthis1->exists();
            break;
         case 6:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( type )");
            *rval = pthis1->type();
            break;
         case 7:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( isDir )");
            *rval = pthis1->isDir();
            break;
         case 8:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( isFile )");
            *rval = pthis1->isFile();
            break;
         case 9:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( isLink )");
            *rval = pthis1->isLink();
            break;
         case 10:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( accessedTime )");
            *rval = pthis1->accessedTime();
            break;
         case 11:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( modifiedTime )");
            *rval = pthis1->modifiedTime();
            break;
         case 12:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( createdTime )");
            *rval = pthis1->createdTime();
            break;
         case 13:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( size )");
            *rval = pthis1->size();
            break;
         case 14:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( path )");
            *rval = pthis1->path();
            break;
         case 15:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( name )");
            *rval = pthis1->name();
            break;
         case 16:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( setName )");
            *rval = pthis1->setName(argv[0].toStr(),&ex);
            break;
         case 17:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( get )");
            *rval = siFs(pthis1->get(argv[0].toStr()));
            break;
         case 18:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( parent )");
            *rval = siFs(pthis1->parent());
            break;
         case 19:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( hasChildren )");
            *rval = pthis1->hasChildren();
            break;
         case 20:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( scandir )");
            *rval = o3_new(tScrVec<Str>)(pthis1->scandir(argv[0].toStr()));
            break;
         case 21:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( children )");
            *rval = o3_new(tScrVec<siFs>)(pthis1->children());
            break;
         case 22:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( createDir )");
            *rval = pthis1->createDir();
            break;
         case 23:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( createFile )");
            *rval = pthis1->createFile();
            break;
         case 24:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( createLink )");
            *rval = pthis1->createLink(siFs (argv[0].toScr()));
            break;
         case 25:
            if (argc > 1)
               return o3_new(cEx)("Invalid argument count. ( remove )");
            *rval = pthis1->remove(argc > 0 ? argv[0].toBool() : true);
            break;
         case 26:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( copy )");
            *rval = siFs(pthis1->copy(siFs (argv[0].toScr()),&ex));
            break;
         case 27:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( move )");
            *rval = siFs(pthis1->move(siFs (argv[0].toScr()),&ex));
            break;
         case 28:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( open )");
            *rval = siStream(pthis1->open(argv[0].toStr(),&ex));
            break;
         case 29:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( canRead )");
            *rval = pthis1->canRead();
            break;
         case 30:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( canWrite )");
            *rval = pthis1->canWrite();
            break;
         case 31:
            *rval = o3_new(cScrBuf)(pthis1->blob());
            break;
         case 32:
            if (argc==1) {
               Var::Type type0 = argv[0].type();
               if (siBuf sibuf = siBuf(argv[0].toScr())) {
                  *rval = o3_new(cScrBuf)(pthis1->setBlob(Buf(siBuf(argv[0].toScr()))));
                  return ex;
               }
               else if(siStream  sistream  = siStream (argv[0].toScr())) {
                  *rval = siStream(pthis1->setBlob(siStream (argv[0].toScr()),&ex));
                  return ex;
               }
               else{
                  return o3_new(cEx)("Invalid argument type.");
               }
            }
            else{
               return o3_new(cEx)("Invalid argument count.");
            }

            break;
         case 34:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( data )");
            *rval = pthis1->data();
            break;
         case 35:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( setData )");
            *rval = pthis1->setData(argv[0].toStr());
            break;
         case 36:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( onchange )");
            *rval = pthis1->onchange();
            break;
         case 37:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( setOnchange )");
            *rval = pthis1->setOnchange(ctx,argv[0].toScr());
            break;
         case 38:
            if (argc != 2)
               return o3_new(cEx)("Invalid argument count. ( fopen )");
            *rval = siStream(pthis1->fopen(argv[0].toStr(),argv[1].toStr()));
            break;
         case 39:
            if (argc != 2)
               return o3_new(cEx)("Invalid argument count. ( fseek )");
            *rval = pthis1->fseek(siStream (argv[0].toScr()),argv[1].toInt32());
            break;
         case 40:
            if (argc != 2)
               return o3_new(cEx)("Invalid argument count. ( fread )");
            *rval = pthis1->fread(siStream (argv[0].toScr()),argv[1].toInt32());
            break;
         case 41:
            if (argc != 2)
               return o3_new(cEx)("Invalid argument count. ( fwrite )");
            *rval = pthis1->fwrite(siStream (argv[0].toScr()),argv[1].toStr());
            break;
         case 42:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( fflush )");
            *rval = pthis1->fflush(siStream (argv[0].toScr()));
            break;
         case 43:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( fclose )");
            *rval = pthis1->fclose(siStream (argv[0].toScr()));
            break;
      }
      return ex;
}

siEx cFs1Base::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
