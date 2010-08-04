#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cXmlNamedNodeMap1::select()
{
   return clsTraits();
}

Trait* cXmlNamedNodeMap1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cXmlNamedNodeMap1",  0,                    0,              0,      cScr::clsTraits()  },
         {      0,      Trait::TYPE_FUN,        "cXmlNamedNodeMap1",  "__query__",          clsInvoke,      0,      0                  },
         {      1,      Trait::TYPE_FUN,        "cXmlNamedNodeMap1",  "__deleter__",        clsInvoke,      1,      0                  },
         {      2,      Trait::TYPE_FUN,        "cXmlNamedNodeMap1",  "__getter__",         clsInvoke,      2,      0                  },
         {      3,      Trait::TYPE_FUN,        "cXmlNamedNodeMap1",  "getNamedItem",       clsInvoke,      3,      0                  },
         {      4,      Trait::TYPE_SET,        "cXmlNamedNodeMap1",  "namedItem",          clsInvoke,      4,      0                  },
         {      5,      Trait::TYPE_FUN,        "cXmlNamedNodeMap1",  "removeNamedItem",    clsInvoke,      5,      0                  },
         {      6,      Trait::TYPE_GET,        "cXmlNamedNodeMap1",  "length",             clsInvoke,      6,      0                  },
         {      0,      Trait::TYPE_END,        "cXmlNamedNodeMap1",  0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cXmlNamedNodeMap1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cXmlNamedNodeMap1",  0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_END,        "cXmlNamedNodeMap1",  0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cXmlNamedNodeMap1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cXmlNamedNodeMap1* pthis1 = (cXmlNamedNodeMap1*) pthis;

      switch(index) {
         case 0:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( __query__ )");
            *rval = pthis1->__query__(argv[0].toInt32());
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
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( getNamedItem )");
            *rval = siXmlNode(pthis1->getNamedItem(ctx,argv[0].toStr()));
            break;
         case 4:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( setNamedItem )");
            *rval = siXmlNode(pthis1->setNamedItem(ctx,siXmlNode (argv[0].toScr())));
            break;
         case 5:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( removeNamedItem )");
            *rval = siXmlNode(pthis1->removeNamedItem(ctx,argv[0].toStr()));
            break;
         case 6:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( length )");
            *rval = pthis1->length();
            break;
      }
      return ex;
}

siEx cXmlNamedNodeMap1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
