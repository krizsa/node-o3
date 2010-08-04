#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cBarcode1::select()
{
   return clsTraits();
}

Trait* cBarcode1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cBarcode1",          0,                    0,              0,      cScr::clsTraits()  },
         {      0,      Trait::TYPE_FUN,        "cBarcode1",          "__self__",           clsInvoke,      0,      0                  },
         {      0,      Trait::TYPE_END,        "cBarcode1",          0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cBarcode1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cBarcode1",          0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_GET,        "cO3",                "barcode",            extInvoke,      0,      0                  },
         {      1,      Trait::TYPE_FUN,        "cImage1",            "scanbarcodes",       extInvoke,      1,      0                  },
         {      0,      Trait::TYPE_END,        "cBarcode1",          0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cBarcode1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cBarcode1* pthis1 = (cBarcode1*) pthis;

      switch(index) {
         case 0:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( __self__ )");
            *rval = pthis1->__self__();
            break;
      }
      return ex;
}

siEx cBarcode1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cBarcode1* pthis1 = (cBarcode1*) pthis;

      switch(index) {
         case 0:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( barcode )");
            *rval = pthis1->barcode(ctx);
            break;
         case 1:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( scanbarcodes )");
            *rval = o3_new(tScrVec<Str>)(pthis1->scanbarcodes(pthis));
            break;
      }
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
