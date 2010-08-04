#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cXmlAttr1::select()
{
   return clsTraits();
}

Trait* cXmlAttr1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cXmlAttr1",          0,                    0,              0,      cXmlNode1::clsTraits()  },
         {      0,      Trait::TYPE_GET,        "cXmlAttr1",          "name",               clsInvoke,      0,      0                  },
         {      1,      Trait::TYPE_GET,        "cXmlAttr1",          "specified",          clsInvoke,      1,      0                  },
         {      2,      Trait::TYPE_GET,        "cXmlAttr1",          "value",              clsInvoke,      2,      0                  },
         {      2,      Trait::TYPE_SET,        "cXmlAttr1",          "value",              clsInvoke,      3,      0                  },
         {      0,      Trait::TYPE_END,        "cXmlAttr1",          0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cXmlAttr1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cXmlAttr1",          0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_END,        "cXmlAttr1",          0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cXmlAttr1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cXmlAttr1* pthis1 = (cXmlAttr1*) pthis;

      switch(index) {
         case 0:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( name )");
            *rval = pthis1->name();
            break;
         case 1:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( specified )");
            *rval = pthis1->specified();
            break;
         case 2:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( value )");
            *rval = pthis1->value();
            break;
         case 3:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( setValue )");
            pthis1->setValue(argv[0].toStr());
            break;
      }
      return ex;
}

siEx cXmlAttr1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
