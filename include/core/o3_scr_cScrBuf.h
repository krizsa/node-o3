#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cScrBuf::select()
{
   return clsTraits();
}

Trait* cScrBuf::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cScrBuf",            0,                    0,              0,      cScr::clsTraits()  },
         {      0,      Trait::TYPE_GET,        "cScrBuf",            "length",             clsInvoke,      0,      0                  },
         {      0,      Trait::TYPE_SET,        "cScrBuf",            "length",             clsInvoke,      1,      0                  },
         {      1,      Trait::TYPE_FUN,        "cScrBuf",            "append",             clsInvoke,      2,      0                  },
         {      2,      Trait::TYPE_FUN,        "cScrBuf",            "slice",              clsInvoke,      3,      0                  },
         {      3,      Trait::TYPE_FUN,        "cScrBuf",            "__enumerator__",     clsInvoke,      4,      0                  },
         {      4,      Trait::TYPE_FUN,        "cScrBuf",            "__query__",          clsInvoke,      5,      0                  },
         {      5,      Trait::TYPE_FUN,        "cScrBuf",            "__getter__",         clsInvoke,      6,      0                  },
         {      6,      Trait::TYPE_FUN,        "cScrBuf",            "__setter__",         clsInvoke,      7,      0                  },
         {      0,      Trait::TYPE_END,        "cScrBuf",            0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cScrBuf::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cScrBuf",            0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_END,        "cScrBuf",            0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cScrBuf::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cScrBuf* pthis1 = (cScrBuf*) pthis;

      switch(index) {
         case 0:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( length )");
            *rval = pthis1->length();
            break;
         case 1:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( setLength )");
            *rval = pthis1->setLength(argv[0].toInt32());
            break;
         case 2:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( append )");
            pthis1->append(Buf(siBuf(argv[0].toScr())));
            break;
         case 3:
            if (argc != 2)
               return o3_new(cEx)("Invalid argument count. ( slice )");
            *rval = o3_new(cScrBuf)(pthis1->slice(argv[0].toInt32(),argv[1].toInt32()));
            break;
         case 4:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( __enumerator__ )");
            *rval = pthis1->__enumerator__(argv[0].toInt32());
            break;
         case 5:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( __query__ )");
            *rval = pthis1->__query__(argv[0].toInt32());
            break;
         case 6:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( __getter__ )");
            *rval = pthis1->__getter__(argv[0].toInt32());
            break;
         case 7:
            if (argc != 2)
               return o3_new(cEx)("Invalid argument count. ( __setter__ )");
            *rval = pthis1->__setter__(argv[0].toInt32(),argv[1].toInt32());
            break;
      }
      return ex;
}

siEx cScrBuf::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
