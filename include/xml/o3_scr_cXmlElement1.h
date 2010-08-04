#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cXmlElement1::select()
{
   return clsTraits();
}

Trait* cXmlElement1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cXmlElement1",       0,                    0,              0,      cXmlNode1::clsTraits()  },
         {      0,      Trait::TYPE_FUN,        "cXmlElement1",       "selectSingleNode",   clsInvoke,      0,      0                  },
         {      1,      Trait::TYPE_FUN,        "cXmlElement1",       "selectNodes",        clsInvoke,      1,      0                  },
         {      2,      Trait::TYPE_GET,        "cXmlElement1",       "tagName",            clsInvoke,      2,      0                  },
         {      3,      Trait::TYPE_FUN,        "cXmlElement1",       "getAttribute",       clsInvoke,      3,      0                  },
         {      4,      Trait::TYPE_FUN,        "cXmlElement1",       "setAttribute",       clsInvoke,      4,      0                  },
         {      5,      Trait::TYPE_FUN,        "cXmlElement1",       "removeAttribute",    clsInvoke,      5,      0                  },
         {      6,      Trait::TYPE_FUN,        "cXmlElement1",       "getAttributeNode",   clsInvoke,      6,      0                  },
         {      7,      Trait::TYPE_FUN,        "cXmlElement1",       "setAttributeNode",   clsInvoke,      7,      0                  },
         {      8,      Trait::TYPE_FUN,        "cXmlElement1",       "removeAttributeNode",clsInvoke,      8,      0                  },
         {      9,      Trait::TYPE_FUN,        "cXmlElement1",       "getElementsByTagName",clsInvoke,      9,      0                  },
         {      10,     Trait::TYPE_FUN,        "cXmlElement1",       "normalize",          clsInvoke,      10,     0                  },
         {      0,      Trait::TYPE_END,        "cXmlElement1",       0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cXmlElement1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cXmlElement1",       0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_END,        "cXmlElement1",       0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cXmlElement1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cXmlElement1* pthis1 = (cXmlElement1*) pthis;

      switch(index) {
         case 0:
            if (argc < 1 && argc > 2)
               return o3_new(cEx)("Invalid argument count. ( selectSingleNode )");
            *rval = siXmlNode(pthis1->selectSingleNode(ctx,argv[0].toStr(),argc > 1 ? argv[1].toScr() : 0));
            break;
         case 1:
            if (argc < 1 && argc > 2)
               return o3_new(cEx)("Invalid argument count. ( selectNodes )");
            *rval = siXmlNodeArray(pthis1->selectNodes(ctx,argv[0].toStr(),argc > 1 ? argv[1].toScr() : 0));
            break;
         case 2:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( tagName )");
            *rval = pthis1->tagName();
            break;
         case 3:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( getAttribute )");
            *rval = pthis1->getAttribute(ctx,argv[0].toStr());
            break;
         case 4:
            if (argc != 2)
               return o3_new(cEx)("Invalid argument count. ( setAttribute )");
            pthis1->setAttribute(ctx,argv[0].toStr(),argv[1].toStr());
            break;
         case 5:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( removeAttribute )");
            pthis1->removeAttribute(ctx,argv[0].toStr());
            break;
         case 6:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( getAttributeNode )");
            *rval = siXmlAttr(pthis1->getAttributeNode(ctx,argv[0].toStr()));
            break;
         case 7:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( setAttributeNode )");
            *rval = siXmlAttr(pthis1->setAttributeNode(ctx,siXmlAttr (argv[0].toScr())));
            break;
         case 8:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( removeAttributeNode )");
            *rval = siXmlAttr(pthis1->removeAttributeNode(siXmlAttr (argv[0].toScr())));
            break;
         case 9:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( getElementsByTagName )");
            *rval = siXmlNodeList(pthis1->getElementsByTagName(argv[0].toStr()));
            break;
         case 10:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( normalize )");
            pthis1->normalize();
            break;
      }
      return ex;
}

siEx cXmlElement1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
