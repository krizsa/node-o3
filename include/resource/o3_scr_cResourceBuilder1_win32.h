#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cResourceBuilder1::select()
{
   return clsTraits();
}

Trait* cResourceBuilder1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cResourceBuilder1",  0,                    0,              0,      cScr::clsTraits()  },
         {      0,      Trait::TYPE_FUN,        "cResourceBuilder1",  "addAsResource",      clsInvoke,      0,      0                  },
         {      1,      Trait::TYPE_FUN,        "cResourceBuilder1",  "buildAndAppend",     clsInvoke,      1,      0                  },
         {      0,      Trait::TYPE_END,        "cResourceBuilder1",  0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cResourceBuilder1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cResourceBuilder1",  0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_FUN,        "cO3",                "resourceBuilder",    extInvoke,      0,      0                  },
         {      1,      Trait::TYPE_FUN,        "cFs1",               "removeResource",     extInvoke,      1,      0                  },
         {      0,      Trait::TYPE_END,        "cResourceBuilder1",  0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cResourceBuilder1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cResourceBuilder1* pthis1 = (cResourceBuilder1*) pthis;

      switch(index) {
         case 0:
            if (argc < 1 && argc > 2)
               return o3_new(cEx)("Invalid argument count. ( addAsResource )");
            *rval = pthis1->addAsResource(siFs (argv[0].toScr()),argc > 1 ? argv[1].toStr() : 0);
            break;
         case 1:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( buildAndAppend )");
            pthis1->buildAndAppend(siFs (argv[0].toScr()),&ex);
            break;
      }
      return ex;
}

siEx cResourceBuilder1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cResourceBuilder1* pthis1 = (cResourceBuilder1*) pthis;

      switch(index) {
         case 0:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( resourceBuilder )");
            *rval = pthis1->resourceBuilder();
            break;
         case 1:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( removeResource )");
            *rval = pthis1->removeResource(siFs (argv[0].toScr()));
            break;
      }
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
