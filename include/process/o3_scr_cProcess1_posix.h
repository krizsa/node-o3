#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cProcess1::select()
{
   return clsTraits();
}

Trait* cProcess1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cProcess1",          0,                    0,              0,      cProcess1Base::clsTraits()  },
         {      0,      Trait::TYPE_END,        "cProcess1",          0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cProcess1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cProcess1",          0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_FUN,        "cO3",                "process",            extInvoke,      0,      0                  },
         {      1,      Trait::TYPE_FUN,        "cO3",                "system",             extInvoke,      1,      0                  },
         {      0,      Trait::TYPE_END,        "cProcess1",          0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cProcess1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      return ex;
}

siEx cProcess1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cProcess1* pthis1 = (cProcess1*) pthis;

      switch(index) {
         case 0:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( process )");
            *rval = pthis1->process();
            break;
         case 1:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( system )");
            *rval = pthis1->system(argv[0].toStr());
            break;
      }
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
