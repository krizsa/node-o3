#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cScreen1::select()
{
   return clsTraits();
}

Trait* cScreen1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cScreen1",           0,                    0,              0,      cScreen1Base::clsTraits()  },
         {      0,      Trait::TYPE_END,        "cScreen1",           0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cScreen1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cScreen1",           0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_FUN,        "cO3",                "screen",             extInvoke,      0,      0                  },
         {      0,      Trait::TYPE_END,        "cScreen1",           0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cScreen1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      return ex;
}

siEx cScreen1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cScreen1* pthis1 = (cScreen1*) pthis;

      switch(index) {
         case 0:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( screen )");
            *rval = pthis1->screen(ctx);
            break;
      }
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
