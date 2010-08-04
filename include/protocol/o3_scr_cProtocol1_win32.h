#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cProtocol1::select()
{
   return clsTraits();
}

Trait* cProtocol1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cProtocol1",         0,                    0,              0,      cScr::clsTraits()  },
         {      0,      Trait::TYPE_FUN,        "cProtocol1",         "addSource",          clsInvoke,      0,      0                  },
         {      0,      Trait::TYPE_END,        "cProtocol1",         0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cProtocol1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cProtocol1",         0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_GET,        "cO3",                "protocolHandler",    extInvoke,      0,      0                  },
         {      0,      Trait::TYPE_END,        "cProtocol1",         0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cProtocol1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cProtocol1* pthis1 = (cProtocol1*) pthis;

      switch(index) {
         case 0:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( addSource )");
            *rval = pthis1->addSource(argv[0].toScr());
            break;
      }
      return ex;
}

siEx cProtocol1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cProtocol1* pthis1 = (cProtocol1*) pthis;

      switch(index) {
         case 0:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( protocolHandler )");
            *rval = siScr(pthis1->protocolHandler(ctx));
            break;
      }
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
