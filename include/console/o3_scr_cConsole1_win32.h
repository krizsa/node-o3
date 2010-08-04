#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cConsole1::select()
{
   return clsTraits();
}

Trait* cConsole1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cConsole1",          0,                    0,              0,      cScr::clsTraits()  },
         {      0,      Trait::TYPE_END,        "cConsole1",          0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cConsole1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cConsole1",          0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_GET,        "cO3",                "stdIn",              extInvoke,      0,      0                  },
         {      1,      Trait::TYPE_GET,        "cO3",                "stdOut",             extInvoke,      1,      0                  },
         {      2,      Trait::TYPE_GET,        "cO3",                "stdErr",             extInvoke,      2,      0                  },
         {      3,      Trait::TYPE_FUN,        "cO3",                "print",              extInvoke,      3,      0                  },
         {      0,      Trait::TYPE_END,        "cConsole1",          0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cConsole1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      return ex;
}

siEx cConsole1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cConsole1* pthis1 = (cConsole1*) pthis;

      switch(index) {
         case 0:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( stdIn )");
            *rval = siStream(pthis1->stdIn(ctx));
            break;
         case 1:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( stdOut )");
            *rval = siStream(pthis1->stdOut(ctx));
            break;
         case 2:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( stdErr )");
            *rval = siStream(pthis1->stdErr(ctx));
            break;
         case 3:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( print )");
            pthis1->print(ctx,argv[0].toStr());
            break;
      }
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
