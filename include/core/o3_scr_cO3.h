#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cLoadProgress::select()
{
   return clsTraits();
}

Trait* cLoadProgress::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cLoadProgress",      0,                    0,              0,      cScr::clsTraits()  },
         {      0,      Trait::TYPE_GET,        "cLoadProgress",      "READY_STATE_UNINITIALIZED",clsInvoke,      0,      0                  },
         {      1,      Trait::TYPE_GET,        "cLoadProgress",      "READY_STATE_LOADING",clsInvoke,      1,      0                  },
         {      2,      Trait::TYPE_GET,        "cLoadProgress",      "READY_STATE_LOADED", clsInvoke,      2,      0                  },
         {      3,      Trait::TYPE_GET,        "cLoadProgress",      "READY_STATE_INTERACTIVE",clsInvoke,      3,      0                  },
         {      4,      Trait::TYPE_GET,        "cLoadProgress",      "READY_STATE_COMPLETED",clsInvoke,      4,      0                  },
         {      5,      Trait::TYPE_GET,        "cLoadProgress",      "bytesReceived",      clsInvoke,      5,      0                  },
         {      6,      Trait::TYPE_GET,        "cLoadProgress",      "readyState",         clsInvoke,      6,      0                  },
         {      7,      Trait::TYPE_GET,        "cLoadProgress",      "fileName",           clsInvoke,      7,      0                  },
         {      0,      Trait::TYPE_END,        "cLoadProgress",      0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cLoadProgress::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cLoadProgress",      0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_END,        "cLoadProgress",      0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cLoadProgress::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cLoadProgress* pthis1 = (cLoadProgress*) pthis;

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
            *rval = 4;
            break;
         case 5:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( bytesReceived )");
            *rval = pthis1->bytesReceived();
            break;
         case 6:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( readyState )");
            *rval = pthis1->readyState();
            break;
         case 7:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( fileName )");
            *rval = pthis1->fileName();
            break;
      }
      return ex;
}

siEx cLoadProgress::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
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


Trait* cO3::select()
{
   return clsTraits();
}

Trait* cO3::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cO3",                0,                    0,              0,      cScr::clsTraits()  },
         {      0,      Trait::TYPE_GET,        "cO3",                "args",               clsInvoke,      0,      0                  },
         {      1,      Trait::TYPE_GET,        "cO3",                "envs",               clsInvoke,      1,      0                  },
         {      2,      Trait::TYPE_FUN,        "cO3",                "wait",               clsInvoke,      2,      0                  },
         {      3,      Trait::TYPE_FUN,        "cO3",                "exit",               clsInvoke,      3,      0                  },
         {      4,      Trait::TYPE_GET,        "cO3",                "versionInfo",        clsInvoke,      4,      0                  },
         {      5,      Trait::TYPE_GET,        "cO3",                "settings",           clsInvoke,      5,      0                  },
         {      5,      Trait::TYPE_SET,        "cO3",                "settings",           clsInvoke,      6,      0                  },
         {      6,      Trait::TYPE_GET,        "cO3",                "settingsURL",        clsInvoke,      7,      0                  },
         {      7,      Trait::TYPE_FUN,        "cO3",                "loadModule",         clsInvoke,      8,      0                  },
         {      8,      Trait::TYPE_FUN,        "cO3",                "require",            clsInvoke,      9,      0                  },
         {      9,      Trait::TYPE_FUN,        "cO3",                "loadModules",        clsInvoke,      10,     0                  },
         {      10,     Trait::TYPE_SET,        "cO3",                "onUpdateNotification",clsInvoke,      11,     0                  },
         {      0,      Trait::TYPE_END,        "cO3",                0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cO3::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cO3",                0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_END,        "cO3",                0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cO3::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cO3* pthis1 = (cO3*) pthis;

      switch(index) {
         case 0:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( args )");
            *rval = o3_new(tScrVec<Str>)(pthis1->args());
            break;
         case 1:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( envs )");
            *rval = o3_new(tScrVec<Str>)(pthis1->envs());
            break;
         case 2:
            if (argc > 1)
               return o3_new(cEx)("Invalid argument count. ( wait )");
            pthis1->wait(ctx,argc > 0 ? argv[0].toInt32() : -1);
            break;
         case 3:
            if (argc > 1)
               return o3_new(cEx)("Invalid argument count. ( exit )");
            pthis1->exit(argc > 0 ? argv[0].toInt32() : 0);
            break;
         case 4:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( versionInfo )");
            *rval = pthis1->versionInfo();
            break;
         case 5:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( settings )");
            *rval = pthis1->settings(ctx);
            break;
         case 6:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( setSettings )");
            *rval = pthis1->setSettings(ctx,argv[0].toStr(),&ex);
            break;
         case 7:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( settingsURL )");
            *rval = pthis1->settingsURL();
            break;
         case 8:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( loadModule )");
            *rval = pthis1->loadModule(ctx,argv[0].toStr());
            break;
         case 9:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( require )");
            pthis1->require(ctx,argv[0].toStr());
            break;
         case 10:
            if (argc < 1 && argc > 3)
               return o3_new(cEx)("Invalid argument count. ( loadModules )");
            pthis1->loadModules(ctx,argv[0].toScr(),argc > 1 ? argv[1].toScr() : 0,argc > 2 ? argv[2].toScr() : 0);
            break;
         case 11:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( setOnUpdateNotification )");
            *rval = pthis1->setOnUpdateNotification(ctx,argv[0].toScr());
            break;
      }
      return ex;
}

siEx cO3::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
