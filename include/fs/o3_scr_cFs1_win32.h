#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cFs1::select()
{
   return clsTraits();
}

Trait* cFs1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cFs1",               0,                    0,              0,      cFs1Base::clsTraits()  },
         {      0,      Trait::TYPE_GET,        "cFs1",               "fullPath",           clsInvoke,      0,      0                  },
         {      1,      Trait::TYPE_FUN,        "cFs1",               "openDoc",            clsInvoke,      1,      0                  },
         {      0,      Trait::TYPE_END,        "cFs1",               0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cFs1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cFs1",               0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_GET,        "cO3",                "fs",                 extInvoke,      0,      0                  },
         {      1,      Trait::TYPE_GET,        "cO3",                "cwd",                extInvoke,      1,      0                  },
         {      2,      Trait::TYPE_GET,        "cO3",                "programFiles",       extInvoke,      2,      0                  },
         {      3,      Trait::TYPE_GET,        "cO3",                "appData",            extInvoke,      3,      0                  },
         {      4,      Trait::TYPE_GET,        "cO3",                "tmpDir",             extInvoke,      4,      0                  },
         {      5,      Trait::TYPE_FUN,        "cO3",                "selectFolder",       extInvoke,      5,      0                  },
         {      6,      Trait::TYPE_FUN,        "cO3",                "openFileDialog",     extInvoke,      6,      0                  },
         {      7,      Trait::TYPE_FUN,        "cO3",                "saveAsFile",         extInvoke,      7,      0                  },
         {      0,      Trait::TYPE_END,        "cFs1",               0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cFs1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cFs1* pthis1 = (cFs1*) pthis;

      switch(index) {
         case 0:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( fullPath )");
            *rval = pthis1->fullPath();
            break;
         case 1:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( openDoc )");
            pthis1->openDoc();
            break;
      }
      return ex;
}

siEx cFs1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cFs1* pthis1 = (cFs1*) pthis;

      switch(index) {
         case 0:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( fs )");
            *rval = siFs(pthis1->fs(ctx));
            break;
         case 1:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( cwd )");
            *rval = siFs(pthis1->cwd());
            break;
         case 2:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( programFiles )");
            *rval = siFs(pthis1->programFiles());
            break;
         case 3:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( appData )");
            *rval = siFs(pthis1->appData());
            break;
         case 4:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( tmpDir )");
            *rval = siFs(pthis1->tmpDir());
            break;
         case 5:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( selectFolder )");
            *rval = siFs(pthis1->selectFolder());
            break;
         case 6:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( openFileDialog )");
            *rval = siFs(pthis1->openFileDialog(argv[0].toStr()));
            break;
         case 7:
            if (argc != 2)
               return o3_new(cEx)("Invalid argument count. ( saveAsFile )");
            *rval = pthis1->saveAsFile(argv[0].toStr(),argv[1].toStr());
            break;
      }
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
