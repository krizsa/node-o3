#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cZip1::select()
{
   return clsTraits();
}

Trait* cZip1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cZip1",              0,                    0,              0,      cScr::clsTraits()  },
         {      0,      Trait::TYPE_FUN,        "cZip1",              "add",                clsInvoke,      0,      0                  },
         {      1,      Trait::TYPE_FUN,        "cZip1",              "zipTo",              clsInvoke,      1,      0                  },
         {      0,      Trait::TYPE_END,        "cZip1",              0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cZip1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cZip1",              0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_FUN,        "cO3",                "zip",                extInvoke,      0,      0                  },
         {      0,      Trait::TYPE_END,        "cZip1",              0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cZip1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cZip1* pthis1 = (cZip1*) pthis;

      switch(index) {
         case 0:
            if (argc < 1 && argc > 2)
               return o3_new(cEx)("Invalid argument count. ( add )");
            *rval = pthis1->add(siFs (argv[0].toScr()),argc > 1 ? argv[1].toStr() : 0);
            break;
         case 1:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( zipTo )");
            *rval = pthis1->zipTo(siFs (argv[0].toScr()),&ex);
            break;
      }
      return ex;
}

siEx cZip1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cZip1* pthis1 = (cZip1*) pthis;

      switch(index) {
         case 0:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( zip )");
            *rval = pthis1->zip();
            break;
      }
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cUnzip1::select()
{
   return clsTraits();
}

Trait* cUnzip1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cUnzip1",            0,                    0,              0,      cScr::clsTraits()  },
         {      0,      Trait::TYPE_FUN,        "cUnzip1",            "openZipFile",        clsInvoke,      0,      0                  },
         {      1,      Trait::TYPE_FUN,        "cUnzip1",            "openZipFromStream",  clsInvoke,      1,      0                  },
         {      2,      Trait::TYPE_GET,        "cUnzip1",            "listFiles",          clsInvoke,      2,      0                  },
         {      3,      Trait::TYPE_FUN,        "cUnzip1",            "get",                clsInvoke,      3,      0                  },
         {      4,      Trait::TYPE_FUN,        "cUnzip1",            "unzip",              clsInvoke,      4,      0                  },
         {      0,      Trait::TYPE_END,        "cUnzip1",            0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cUnzip1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cUnzip1",            0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_FUN,        "cO3",                "unzip",              extInvoke,      0,      0                  },
         {      0,      Trait::TYPE_END,        "cUnzip1",            0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cUnzip1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cUnzip1* pthis1 = (cUnzip1*) pthis;

      switch(index) {
         case 0:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( openZipFile )");
            *rval = pthis1->openZipFile(siFs (argv[0].toScr()),&ex);
            break;
         case 1:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( openZipFromStream )");
            *rval = pthis1->openZipFromStream(siStream (argv[0].toScr()),&ex);
            break;
         case 2:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( listFiles )");
            *rval = o3_new(tScrVec<Str>)(pthis1->listFiles());
            break;
         case 3:
            if (argc != 2)
               return o3_new(cEx)("Invalid argument count. ( get )");
            *rval = pthis1->get(argv[0].toStr(),siStream (argv[1].toScr()),&ex);
            break;
         case 4:
            if (argc != 2)
               return o3_new(cEx)("Invalid argument count. ( unzip )");
            pthis1->unzip(siFs (argv[0].toScr()),siFs (argv[1].toScr()));
            break;
      }
      return ex;
}

siEx cUnzip1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cUnzip1* pthis1 = (cUnzip1*) pthis;

      switch(index) {
         case 0:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( unzip )");
            *rval = pthis1->unzip();
            break;
      }
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
