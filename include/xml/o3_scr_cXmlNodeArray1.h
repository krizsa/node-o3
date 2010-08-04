#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cXmlNodeArray1::select()
{
   return clsTraits();
}

Trait* cXmlNodeArray1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cXmlNodeArray1",     0,                    0,              0,      cScr::clsTraits()  },
         {      0,      Trait::TYPE_FUN,        "cXmlNodeArray1",     "__query__",          clsInvoke,      0,      0                  },
         {      1,      Trait::TYPE_FUN,        "cXmlNodeArray1",     "__getter__",         clsInvoke,      1,      0                  },
         {      2,      Trait::TYPE_GET,        "cXmlNodeArray1",     "length",             clsInvoke,      2,      0                  },
         {      0,      Trait::TYPE_END,        "cXmlNodeArray1",     0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cXmlNodeArray1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cXmlNodeArray1",     0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_END,        "cXmlNodeArray1",     0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cXmlNodeArray1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cXmlNodeArray1* pthis1 = (cXmlNodeArray1*) pthis;

      switch(index) {
         case 0:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( __query__ )");
            *rval = pthis1->__query__(argv[0].toInt32());
            break;
         case 1:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( __getter__ )");
            *rval = siXmlNode(pthis1->__getter__(ctx,argv[0].toInt32(),&ex));
            break;
         case 2:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( length )");
            *rval = pthis1->length();
            break;
      }
      return ex;
}

siEx cXmlNodeArray1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
