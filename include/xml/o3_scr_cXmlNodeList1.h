#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cXmlNodeList1::select()
{
   return clsTraits();
}

Trait* cXmlNodeList1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cXmlNodeList1",      0,                    0,              0,      cScr::clsTraits()  },
         {      0,      Trait::TYPE_FUN,        "cXmlNodeList1",      "__query__",          clsInvoke,      0,      0                  },
         {      1,      Trait::TYPE_FUN,        "cXmlNodeList1",      "__deleter__",        clsInvoke,      1,      0                  },
         {      2,      Trait::TYPE_FUN,        "cXmlNodeList1",      "__getter__",         clsInvoke,      2,      0                  },
         {      3,      Trait::TYPE_GET,        "cXmlNodeList1",      "length",             clsInvoke,      3,      0                  },
         {      0,      Trait::TYPE_END,        "cXmlNodeList1",      0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cXmlNodeList1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cXmlNodeList1",      0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_END,        "cXmlNodeList1",      0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cXmlNodeList1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cXmlNodeList1* pthis1 = (cXmlNodeList1*) pthis;

      switch(index) {
         case 0:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( __query__ )");
            *rval = pthis1->__query__(ctx,argv[0].toInt32());
            break;
         case 1:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( __deleter__ )");
            *rval = pthis1->__deleter__(argv[0].toInt32(),&ex);
            break;
         case 2:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( __getter__ )");
            *rval = siXmlNode(pthis1->__getter__(ctx,argv[0].toInt32(),&ex));
            break;
         case 3:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( length )");
            *rval = pthis1->length(ctx);
            break;
      }
      return ex;
}

siEx cXmlNodeList1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
