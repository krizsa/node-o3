#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cJs1::select()
{
   return clsTraits();
}

Trait* cJs1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cJs1",               0,                    0,              0,      cJs1Base::clsTraits()  },
         {      0,      Trait::TYPE_FUN,        "cJs1",               "eval",               clsInvoke,      0,      0                  },
         {      0,      Trait::TYPE_END,        "cJs1",               0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cJs1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cJs1",               0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_GET,        "cO3",                "js",                 extInvoke,      0,      0                  },
         {      0,      Trait::TYPE_END,        "cJs1",               0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cJs1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cJs1* pthis1 = (cJs1*) pthis;

      switch(index) {
         case 0:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( eval )");
            *rval = pthis1->eval(argv[0].toStr(),&ex);
            break;
      }
      return ex;
}

siEx cJs1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cJs1* pthis1 = (cJs1*) pthis;

      switch(index) {
         case 0:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( js )");
            *rval = pthis1->js(ctx);
            break;
      }
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
