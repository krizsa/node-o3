#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cMD5Hash1::select()
{
   return clsTraits();
}

Trait* cMD5Hash1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cMD5Hash1",          0,                    0,              0,      cScr::clsTraits()  },
         {      0,      Trait::TYPE_FUN,        "cMD5Hash1",          "hash",               clsInvoke,      0,      0                  },
         {      0,      Trait::TYPE_FUN,        "cMD5Hash1",          "hash",               clsInvoke,      1,      0                  },
         {      0,      Trait::TYPE_END,        "cMD5Hash1",          0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cMD5Hash1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cMD5Hash1",          0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_GET,        "cO3",                "md5",                extInvoke,      0,      0                  },
         {      0,      Trait::TYPE_END,        "cMD5Hash1",          0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cMD5Hash1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cMD5Hash1* pthis1 = (cMD5Hash1*) pthis;

      switch(index) {
         case 0:
            if (argc==1) {
               Var::Type type0 = argv[0].type();
               if (siBuf sibuf = siBuf(argv[0].toScr())) {
                  *rval = o3_new(cScrBuf)(pthis1->hash(Buf(siBuf(argv[0].toScr()))));
                  return ex;
               }
               else if(Var::TYPE_VOID <= type0 && Var::TYPE_SCR >= type0) {
                  *rval = o3_new(cScrBuf)(pthis1->hash(argv[0].toStr()));
               }
               else{
                  return o3_new(cEx)("Invalid argument type.");
               }
            }
            else{
               return o3_new(cEx)("Invalid argument count.");
            }
            break;
      }
      return ex;
}

siEx cMD5Hash1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cMD5Hash1* pthis1 = (cMD5Hash1*) pthis;

      switch(index) {
         case 0:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( md5 )");
            *rval = pthis1->md5(ctx);
            break;
      }
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
