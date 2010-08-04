#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cJs1Base::select()
{
   return clsTraits();
}

Trait* cJs1Base::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cJs1Base",           0,                    0,              0,      cScr::clsTraits()  },
         {      0,      Trait::TYPE_FUN,        "cJs1Base",           "eval",               clsInvoke,      0,      0                  },
         {      0,      Trait::TYPE_END,        "cJs1Base",           0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cJs1Base::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cJs1Base",           0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_END,        "cJs1Base",           0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cJs1Base::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cJs1Base* pthis1 = (cJs1Base*) pthis;

      switch(index) {
         case 0:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( eval )");
            *rval = pthis1->eval(argv[0].toStr(),&ex);
            break;
      }
      return ex;
}

siEx cJs1Base::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
