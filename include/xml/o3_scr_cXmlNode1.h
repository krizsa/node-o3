#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cXmlNode1::select()
{
   return clsTraits();
}

Trait* cXmlNode1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cXmlNode1",          0,                    0,              0,      cScr::clsTraits()  },
         {      0,      Trait::TYPE_GET,        "cXmlNode1",          "ELEMENT",            clsInvoke,      0,      0                  },
         {      1,      Trait::TYPE_GET,        "cXmlNode1",          "ATTRIBUTE",          clsInvoke,      1,      0                  },
         {      2,      Trait::TYPE_GET,        "cXmlNode1",          "TEXT",               clsInvoke,      2,      0                  },
         {      3,      Trait::TYPE_GET,        "cXmlNode1",          "CDATA_SECTION",      clsInvoke,      3,      0                  },
         {      4,      Trait::TYPE_GET,        "cXmlNode1",          "COMMENT",            clsInvoke,      4,      0                  },
         {      5,      Trait::TYPE_GET,        "cXmlNode1",          "DOCUMENT",           clsInvoke,      5,      0                  },
         {      6,      Trait::TYPE_FUN,        "cXmlNode1",          "replaceNode",        clsInvoke,      6,      0                  },
         {      7,      Trait::TYPE_GET,        "cXmlNode1",          "xml",                clsInvoke,      7,      0                  },
         {      8,      Trait::TYPE_GET,        "cXmlNode1",          "nodeName",           clsInvoke,      8,      0                  },
         {      9,      Trait::TYPE_GET,        "cXmlNode1",          "nodeValue",          clsInvoke,      9,      0                  },
         {      9,      Trait::TYPE_SET,        "cXmlNode1",          "nodeValue",          clsInvoke,      10,     0                  },
         {      10,     Trait::TYPE_GET,        "cXmlNode1",          "nodeType",           clsInvoke,      11,     0                  },
         {      11,     Trait::TYPE_GET,        "cXmlNode1",          "parentNode",         clsInvoke,      12,     0                  },
         {      12,     Trait::TYPE_GET,        "cXmlNode1",          "childNodes",         clsInvoke,      13,     0                  },
         {      13,     Trait::TYPE_GET,        "cXmlNode1",          "firstChild",         clsInvoke,      14,     0                  },
         {      14,     Trait::TYPE_GET,        "cXmlNode1",          "lastChild",          clsInvoke,      15,     0                  },
         {      15,     Trait::TYPE_GET,        "cXmlNode1",          "previousSibling",    clsInvoke,      16,     0                  },
         {      16,     Trait::TYPE_GET,        "cXmlNode1",          "nextSibling",        clsInvoke,      17,     0                  },
         {      17,     Trait::TYPE_GET,        "cXmlNode1",          "attributes",         clsInvoke,      18,     0                  },
         {      18,     Trait::TYPE_GET,        "cXmlNode1",          "ownerDocument",      clsInvoke,      19,     0                  },
         {      19,     Trait::TYPE_FUN,        "cXmlNode1",          "insertBefore",       clsInvoke,      20,     0                  },
         {      20,     Trait::TYPE_FUN,        "cXmlNode1",          "replaceChild",       clsInvoke,      21,     0                  },
         {      21,     Trait::TYPE_FUN,        "cXmlNode1",          "removeChild",        clsInvoke,      22,     0                  },
         {      22,     Trait::TYPE_FUN,        "cXmlNode1",          "appendChild",        clsInvoke,      23,     0                  },
         {      23,     Trait::TYPE_GET,        "cXmlNode1",          "hasChildNodes",      clsInvoke,      24,     0                  },
         {      24,     Trait::TYPE_FUN,        "cXmlNode1",          "cloneNode",          clsInvoke,      25,     0                  },
         {      25,     Trait::TYPE_GET,        "cXmlNode1",          "namespaceURI",       clsInvoke,      26,     0                  },
         {      0,      Trait::TYPE_END,        "cXmlNode1",          0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cXmlNode1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cXmlNode1",          0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_END,        "cXmlNode1",          0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cXmlNode1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cXmlNode1* pthis1 = (cXmlNode1*) pthis;

      switch(index) {
         case 0:
            *rval = 1;
            break;
         case 1:
            *rval = 2;
            break;
         case 2:
            *rval = 3;
            break;
         case 3:
            *rval = 4;
            break;
         case 4:
            *rval = 8;
            break;
         case 5:
            *rval = 9;
            break;
         case 6:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( replaceNode )");
            *rval = siXmlNode(pthis1->replaceNode(ctx,siXmlNode (argv[0].toScr())));
            break;
         case 7:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( xml )");
            *rval = pthis1->xml();
            break;
         case 8:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( nodeName )");
            *rval = pthis1->nodeName();
            break;
         case 9:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( nodeValue )");
            *rval = pthis1->nodeValue();
            break;
         case 10:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( setNodeValue )");
            pthis1->setNodeValue(argv[0].toStr());
            break;
         case 11:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( nodeType )");
            *rval = pthis1->nodeType();
            break;
         case 12:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( parentNode )");
            *rval = siXmlNode(pthis1->parentNode(ctx));
            break;
         case 13:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( childNodes )");
            *rval = siXmlNodeList(pthis1->childNodes());
            break;
         case 14:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( firstChild )");
            *rval = siXmlNode(pthis1->firstChild(ctx));
            break;
         case 15:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( lastChild )");
            *rval = siXmlNode(pthis1->lastChild(ctx));
            break;
         case 16:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( previousSibling )");
            *rval = siXmlNode(pthis1->previousSibling(ctx));
            break;
         case 17:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( nextSibling )");
            *rval = siXmlNode(pthis1->nextSibling(ctx));
            break;
         case 18:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( attributes )");
            *rval = siXmlNamedNodeMap(pthis1->attributes());
            break;
         case 19:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( ownerDocument )");
            *rval = siXmlDocument(pthis1->ownerDocument(ctx));
            break;
         case 20:
            if (argc != 2)
               return o3_new(cEx)("Invalid argument count. ( insertBefore )");
            *rval = siXmlNode(pthis1->insertBefore(siXmlNode (argv[0].toScr()),siXmlNode (argv[1].toScr())));
            break;
         case 21:
            if (argc != 2)
               return o3_new(cEx)("Invalid argument count. ( replaceChild )");
            *rval = siXmlNode(pthis1->replaceChild(siXmlNode (argv[0].toScr()),siXmlNode (argv[1].toScr())));
            break;
         case 22:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( removeChild )");
            *rval = siXmlNode(pthis1->removeChild(siXmlNode (argv[0].toScr())));
            break;
         case 23:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( appendChild )");
            *rval = siXmlNode(pthis1->appendChild(ctx,siXmlNode (argv[0].toScr())));
            break;
         case 24:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( hasChildNodes )");
            *rval = pthis1->hasChildNodes();
            break;
         case 25:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( cloneNode )");
            *rval = siXmlNode(pthis1->cloneNode(ctx,argv[0].toBool()));
            break;
         case 26:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( namespaceURI )");
            *rval = pthis1->namespaceURI();
            break;
      }
      return ex;
}

siEx cXmlNode1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
