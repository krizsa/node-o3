#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cBlob1::select()
{
   return clsTraits();
}

Trait* cBlob1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cBlob1",             0,                    0,              0,      cScr::clsTraits()  },
         {      0,      Trait::TYPE_FUN,        "cBlob1",             "__self__",           clsInvoke,      0,      0                  },
         {      0,      Trait::TYPE_FUN,        "cBlob1",             "__self__",           clsInvoke,      1,      0                  },
         {      0,      Trait::TYPE_FUN,        "cBlob1",             "__self__",           clsInvoke,      2,      0                  },
         {      1,      Trait::TYPE_FUN,        "cBlob1",             "fromString",         clsInvoke,      3,      0                  },
         {      2,      Trait::TYPE_FUN,        "cBlob1",             "fromHex",            clsInvoke,      4,      0                  },
         {      3,      Trait::TYPE_FUN,        "cBlob1",             "fromBase64",         clsInvoke,      5,      0                  },
         {      0,      Trait::TYPE_END,        "cBlob1",             0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cBlob1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cBlob1",             0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_GET,        "cO3",                "blob",               extInvoke,      0,      0                  },
         {      1,      Trait::TYPE_FUN,        "cScrBuf",            "toString",           extInvoke,      1,      0                  },
         {      2,      Trait::TYPE_FUN,        "cScrBuf",            "toHex",              extInvoke,      2,      0                  },
         {      3,      Trait::TYPE_FUN,        "cScrBuf",            "toBase64",           extInvoke,      3,      0                  },
         {      4,      Trait::TYPE_FUN,        "cScrBuf",            "replace",            extInvoke,      4,      0                  },
         {      4,      Trait::TYPE_FUN,        "cScrBuf",            "replace",            extInvoke,      5,      0                  },
         {      5,      Trait::TYPE_FUN,        "cScrBuf",            "replaceUtf16",       extInvoke,      6,      0                  },
         {      0,      Trait::TYPE_END,        "cBlob1",             0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cBlob1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cBlob1* pthis1 = (cBlob1*) pthis;

      switch(index) {
         case 0:
            if (argc==0) {
               *rval = o3_new(cScrBuf)(pthis1->__self__(ctx));
            }
            else if(argc==1) {
               Var::Type type0 = argv[0].type();
               if (Var::TYPE_VOID <= type0 && Var::TYPE_INT32 >= type0) {
                  *rval = o3_new(cScrBuf)(pthis1->__self__(ctx,argv[0].toInt32()));
               }
               else if(Var::TYPE_INT64 <= type0 && Var::TYPE_SCR >= type0) {
                  *rval = o3_new(cScrBuf)(pthis1->__self__(argv[0].toStr()));
               }
               else{
                  return o3_new(cEx)("Invalid argument type.");
               }
            }
            else{
               return o3_new(cEx)("Invalid argument count.");
            }
            break;
         case 3:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( fromString )");
            *rval = o3_new(cScrBuf)(pthis1->fromString(argv[0].toStr()));
            break;
         case 4:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( fromHex )");
            *rval = o3_new(cScrBuf)(pthis1->fromHex(argv[0].toStr()));
            break;
         case 5:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( fromBase64 )");
            *rval = o3_new(cScrBuf)(pthis1->fromBase64(argv[0].toStr()));
            break;
      }
      return ex;
}

siEx cBlob1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cBlob1* pthis1 = (cBlob1*) pthis;

      switch(index) {
         case 0:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( blob )");
            *rval = pthis1->blob(ctx);
            break;
         case 1:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( toString )");
            *rval = pthis1->toString(pthis);
            break;
         case 2:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( toHex )");
            *rval = pthis1->toHex(pthis);
            break;
         case 3:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( toBase64 )");
            *rval = pthis1->toBase64(pthis);
            break;
         case 4:
            if (argc==2) {
               Var::Type type0 = argv[0].type();
               if (siBuf  sibuf  = siBuf (argv[0].toScr())) {
                  pthis1->replace(pthis,siBuf (argv[0].toScr()),siBuf (argv[1].toScr()));
                  return ex;
               }
               else if(Var::TYPE_VOID <= type0 && Var::TYPE_SCR >= type0) {
                  pthis1->replace(pthis,argv[0].toStr(),argv[1].toStr());
               }
               else{
                  return o3_new(cEx)("Invalid argument type.");
               }
            }
            else{
               return o3_new(cEx)("Invalid argument count.");
            }
            break;
         case 6:
            if (argc != 2)
               return o3_new(cEx)("Invalid argument count. ( replaceUtf16 )");
            pthis1->replaceUtf16(pthis,argv[0].toWStr(),argv[1].toWStr());
            break;
      }
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
