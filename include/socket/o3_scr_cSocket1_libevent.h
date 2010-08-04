#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cSocket1::select()
{
   return clsTraits();
}

Trait* cSocket1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cSocket1",           0,                    0,              0,      cSocket1Base::clsTraits()  },
         {      0,      Trait::TYPE_END,        "cSocket1",           0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cSocket1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cSocket1",           0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_FUN,        "cO3",                "socketUDP",          extInvoke,      0,      0                  },
         {      1,      Trait::TYPE_FUN,        "cO3",                "socketTCP",          extInvoke,      1,      0                  },
         {      0,      Trait::TYPE_END,        "cSocket1",           0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cSocket1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      return ex;
}

siEx cSocket1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cSocket1* pthis1 = (cSocket1*) pthis;

      switch(index) {
         case 0:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( socketUDP )");
            *rval = siSocket(pthis1->socketUDP(ctx));
            break;
         case 1:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( socketTCP )");
            *rval = siSocket(pthis1->socketTCP(ctx));
            break;
      }
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
