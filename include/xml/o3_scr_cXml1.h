#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cXml1::select()
{
   return clsTraits();
}

Trait* cXml1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cXml1",              0,                    0,              0,      cScr::clsTraits()  },
         {      0,      Trait::TYPE_FUN,        "cXml1",              "parseFromString",    clsInvoke,      0,      0                  },
         {      0,      Trait::TYPE_END,        "cXml1",              0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cXml1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cXml1",              0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_GET,        "cO3",                "xml",                extInvoke,      0,      0                  },
         {      0,      Trait::TYPE_END,        "cXml1",              0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cXml1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cXml1* pthis1 = (cXml1*) pthis;

      switch(index) {
         case 0:
            if (argc < 1 && argc > 2)
               return o3_new(cEx)("Invalid argument count. ( parseFromString )");
            *rval = siXmlNode(pthis1->parseFromString(ctx,argv[0].toStr(),argc > 1 ? argv[1].toStr() : "text/xml"));
            break;
      }
      return ex;
}

siEx cXml1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cXml1* pthis1 = (cXml1*) pthis;

      switch(index) {
         case 0:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( xml )");
            *rval = siXml(pthis1->xml(ctx));
            break;
      }
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
