#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cResource1::select()
{
   return clsTraits();
}

Trait* cResource1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cResource1",         0,                    0,              0,      cScr::clsTraits()  },
         {      0,      Trait::TYPE_FUN,        "cResource1",         "unpack",             clsInvoke,      0,      0                  },
         {      1,      Trait::TYPE_FUN,        "cResource1",         "list",               clsInvoke,      1,      0                  },
         {      2,      Trait::TYPE_FUN,        "cResource1",         "get",                clsInvoke,      2,      0                  },
         {      3,      Trait::TYPE_FUN,        "cResource1",         "protocolOpen",       clsInvoke,      3,      0                  },
         {      0,      Trait::TYPE_END,        "cResource1",         0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cResource1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cResource1",         0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_GET,        "cO3",                "resources",          extInvoke,      0,      0                  },
         {      0,      Trait::TYPE_END,        "cResource1",         0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cResource1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cResource1* pthis1 = (cResource1*) pthis;

      switch(index) {
         case 0:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( unpack )");
            pthis1->unpack(ctx,siFs (argv[0].toScr()),&ex);
            break;
         case 1:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( list )");
            *rval = o3_new(tScrVec<Str>)(pthis1->list());
            break;
         case 2:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( get )");
            *rval = o3_new(cScrBuf)(pthis1->get(argv[0].toStr()));
            break;
         case 3:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( protocolOpen )");
            *rval = siStream(pthis1->protocolOpen(argv[0].toStr()));
            break;
      }
      return ex;
}

siEx cResource1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cResource1* pthis1 = (cResource1*) pthis;

      switch(index) {
         case 0:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( resources )");
            *rval = pthis1->resources(ctx);
            break;
      }
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
