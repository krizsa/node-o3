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
         {      0,      Trait::TYPE_END,        "cFs1",               0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cFs1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
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
            *rval = pthis1->fs(ctx);
            break;
         case 1:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( cwd )");
            *rval = pthis1->cwd(ctx);
            break;
      }
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
