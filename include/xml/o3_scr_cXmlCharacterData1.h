#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cXmlCharacterData1::select()
{
   return clsTraits();
}

Trait* cXmlCharacterData1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cXmlCharacterData1", 0,                    0,              0,      cXmlNode1::clsTraits()  },
         {      0,      Trait::TYPE_GET,        "cXmlCharacterData1", "data",               clsInvoke,      0,      0                  },
         {      0,      Trait::TYPE_SET,        "cXmlCharacterData1", "data",               clsInvoke,      1,      0                  },
         {      1,      Trait::TYPE_GET,        "cXmlCharacterData1", "length",             clsInvoke,      2,      0                  },
         {      2,      Trait::TYPE_FUN,        "cXmlCharacterData1", "substringData",      clsInvoke,      3,      0                  },
         {      3,      Trait::TYPE_FUN,        "cXmlCharacterData1", "appendData",         clsInvoke,      4,      0                  },
         {      4,      Trait::TYPE_FUN,        "cXmlCharacterData1", "insertData",         clsInvoke,      5,      0                  },
         {      5,      Trait::TYPE_FUN,        "cXmlCharacterData1", "deleteData",         clsInvoke,      6,      0                  },
         {      6,      Trait::TYPE_FUN,        "cXmlCharacterData1", "replaceData",        clsInvoke,      7,      0                  },
         {      0,      Trait::TYPE_END,        "cXmlCharacterData1", 0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cXmlCharacterData1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cXmlCharacterData1", 0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_END,        "cXmlCharacterData1", 0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cXmlCharacterData1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cXmlCharacterData1* pthis1 = (cXmlCharacterData1*) pthis;

      switch(index) {
         case 0:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( data )");
            *rval = pthis1->data();
            break;
         case 1:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( setData )");
            pthis1->setData(argv[0].toStr());
            break;
         case 2:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( length )");
            *rval = pthis1->length();
            break;
         case 3:
            if (argc != 2)
               return o3_new(cEx)("Invalid argument count. ( substringData )");
            *rval = pthis1->substringData(argv[0].toInt32(),argv[1].toInt32(),&ex);
            break;
         case 4:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( appendData )");
            pthis1->appendData(argv[0].toStr());
            break;
         case 5:
            if (argc != 2)
               return o3_new(cEx)("Invalid argument count. ( insertData )");
            pthis1->insertData(argv[0].toInt32(),argv[1].toStr(),&ex);
            break;
         case 6:
            if (argc != 2)
               return o3_new(cEx)("Invalid argument count. ( deleteData )");
            pthis1->deleteData(argv[0].toInt32(),argv[1].toInt32(),&ex);
            break;
         case 7:
            if (argc != 3)
               return o3_new(cEx)("Invalid argument count. ( replaceData )");
            pthis1->replaceData(argv[0].toInt32(),argv[1].toInt32(),argv[2].toStr(),&ex);
            break;
      }
      return ex;
}

siEx cXmlCharacterData1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
