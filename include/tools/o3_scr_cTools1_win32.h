#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cTools1::select()
{
   return clsTraits();
}

Trait* cTools1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cTools1",            0,                    0,              0,      cScr::clsTraits()  },
         {      0,      Trait::TYPE_END,        "cTools1",            0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cTools1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cTools1",            0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_GET,        "cO3",                "tempPath",           extInvoke,      0,      0                  },
         {      1,      Trait::TYPE_GET,        "cO3",                "selfPath",           extInvoke,      1,      0                  },
         {      2,      Trait::TYPE_FUN,        "cO3",                "checkIfInstalled",   extInvoke,      2,      0                  },
         {      3,      Trait::TYPE_FUN,        "cO3",                "regDll",             extInvoke,      3,      0                  },
         {      4,      Trait::TYPE_FUN,        "cO3",                "unregDll",           extInvoke,      4,      0                  },
         {      5,      Trait::TYPE_FUN,        "cO3",                "regUninstaller",     extInvoke,      5,      0                  },
         {      6,      Trait::TYPE_FUN,        "cO3",                "unregUninstaller",   extInvoke,      6,      0                  },
         {      7,      Trait::TYPE_FUN,        "cO3",                "getUninstPath",      extInvoke,      7,      0                  },
         {      8,      Trait::TYPE_FUN,        "cO3",                "regMozillaPlugin",   extInvoke,      8,      0                  },
         {      9,      Trait::TYPE_FUN,        "cO3",                "unregMozillaPlugin", extInvoke,      9,      0                  },
         {      10,     Trait::TYPE_GET,        "cO3",                "adminUser",          extInvoke,      10,     0                  },
         {      11,     Trait::TYPE_GET,        "cO3",                "winVersionMajor",    extInvoke,      11,     0                  },
         {      12,     Trait::TYPE_SET,        "cO3",                "exitCode",           extInvoke,      12,     0                  },
         {      0,      Trait::TYPE_END,        "cTools1",            0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cTools1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      return ex;
}

siEx cTools1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cTools1* pthis1 = (cTools1*) pthis;

      switch(index) {
         case 0:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( tempPath )");
            *rval = pthis1->tempPath();
            break;
         case 1:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( selfPath )");
            *rval = pthis1->selfPath();
            break;
         case 2:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( checkIfInstalled )");
            *rval = pthis1->checkIfInstalled(argv[0].toStr());
            break;
         case 3:
            if (argc != 2)
               return o3_new(cEx)("Invalid argument count. ( regDll )");
            *rval = pthis1->regDll(argv[0].toStr(),argv[1].toBool());
            break;
         case 4:
            if (argc != 2)
               return o3_new(cEx)("Invalid argument count. ( unregDll )");
            *rval = pthis1->unregDll(argv[0].toStr(),argv[1].toBool());
            break;
         case 5:
            if (argc != 2)
               return o3_new(cEx)("Invalid argument count. ( regUninstaller )");
            *rval = pthis1->regUninstaller(ctx,argv[0].toBool(),argv[1].toScr());
            break;
         case 6:
            if (argc != 2)
               return o3_new(cEx)("Invalid argument count. ( unregUninstaller )");
            *rval = pthis1->unregUninstaller(argv[0].toBool(),argv[1].toStr());
            break;
         case 7:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( getUninstPath )");
            *rval = pthis1->getUninstPath(argv[0].toStr());
            break;
         case 8:
            if (argc != 2)
               return o3_new(cEx)("Invalid argument count. ( regMozillaPlugin )");
            *rval = pthis1->regMozillaPlugin(ctx,argv[0].toBool(),argv[1].toScr());
            break;
         case 9:
            if (argc != 4)
               return o3_new(cEx)("Invalid argument count. ( unregMozillaPlugin )");
            *rval = pthis1->unregMozillaPlugin(argv[0].toBool(),argv[1].toStr(),argv[2].toStr(),argv[3].toStr());
            break;
         case 10:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( adminUser )");
            *rval = pthis1->adminUser();
            break;
         case 11:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( winVersionMajor )");
            *rval = pthis1->winVersionMajor();
            break;
         case 12:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( exitCode )");
            *rval = pthis1->exitCode(ctx,argv[0].toInt32());
            break;
      }
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
