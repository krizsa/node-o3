#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cWindow1::select()
{
   return clsTraits();
}

Trait* cWindow1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cWindow1",           0,                    0,              0,      cWindow1Base::clsTraits()  },
         {      0,      Trait::TYPE_END,        "cWindow1",           0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cWindow1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cWindow1",           0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_FUN,        "cO3",                "window",             extInvoke,      0,      0                  },
         {      0,      Trait::TYPE_END,        "cWindow1",           0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cWindow1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      return ex;
}

siEx cWindow1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cWindow1* pthis1 = (cWindow1*) pthis;

      switch(index) {
         case 0:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( window )");
            *rval = pthis1->window(ctx);
            break;
      }
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
