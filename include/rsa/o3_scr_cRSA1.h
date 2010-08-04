#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cRSA1::select()
{
   return clsTraits();
}

Trait* cRSA1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cRSA1",              0,                    0,              0,      cScr::clsTraits()  },
         {      0,      Trait::TYPE_FUN,        "cRSA1",              "encrypt",            clsInvoke,      0,      0                  },
         {      1,      Trait::TYPE_FUN,        "cRSA1",              "decrypt",            clsInvoke,      1,      0                  },
         {      0,      Trait::TYPE_END,        "cRSA1",              0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cRSA1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cRSA1",              0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_GET,        "cO3",                "rsa",                extInvoke,      0,      0                  },
         {      0,      Trait::TYPE_END,        "cRSA1",              0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cRSA1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cRSA1* pthis1 = (cRSA1*) pthis;

      switch(index) {
         case 0:
            if (argc < 3 && argc > 4)
               return o3_new(cEx)("Invalid argument count. ( encrypt )");
            *rval = o3_new(cScrBuf)(pthis1->encrypt(Buf(siBuf(argv[0].toScr())),Buf(siBuf(argv[1].toScr())),Buf(siBuf(argv[2].toScr())),argc > 3 ? argv[3].toBool() : true));
            break;
         case 1:
            if (argc < 3 && argc > 4)
               return o3_new(cEx)("Invalid argument count. ( decrypt )");
            *rval = o3_new(cScrBuf)(pthis1->decrypt(Buf(siBuf(argv[0].toScr())),Buf(siBuf(argv[1].toScr())),Buf(siBuf(argv[2].toScr())),argc > 3 ? argv[3].toBool() : true));
            break;
      }
      return ex;
}

siEx cRSA1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cRSA1* pthis1 = (cRSA1*) pthis;

      switch(index) {
         case 0:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( rsa )");
            *rval = pthis1->rsa(ctx);
            break;
      }
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
