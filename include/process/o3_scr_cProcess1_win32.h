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
         {      0,      Trait::TYPE_FUN,        "cProcess1",          "run",                clsInvoke,      0,      0                  },
         {      1,      Trait::TYPE_FUN,        "cProcess1",          "runSelf",            clsInvoke,      1,      0                  },
         {      2,      Trait::TYPE_FUN,        "cProcess1",          "runSelfElevated",    clsInvoke,      2,      0                  },
         {      3,      Trait::TYPE_FUN,        "cProcess1",          "runSimple",          clsInvoke,      3,      0                  },
         {      4,      Trait::TYPE_GET,        "cProcess1",          "valid",              clsInvoke,      4,      0                  },
         {      5,      Trait::TYPE_GET,        "cProcess1",          "pid",                clsInvoke,      5,      0                  },
         {      6,      Trait::TYPE_FUN,        "cProcess1",          "send",               clsInvoke,      6,      0                  },
         {      7,      Trait::TYPE_FUN,        "cProcess1",          "kill",               clsInvoke,      7,      0                  },
         {      8,      Trait::TYPE_GET,        "cProcess1",          "exitCode",           clsInvoke,      8,      0                  },
         {      0,      Trait::TYPE_END,        "cProcess1",          0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cProcess1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cProcess1",          0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_FUN,        "cO3",                "process",            extInvoke,      0,      0                  },
         {      0,      Trait::TYPE_END,        "cProcess1",          0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cProcess1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cProcess1* pthis1 = (cProcess1*) pthis;

      switch(index) {
         case 0:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( run )");
            *rval = pthis1->run(ctx,argv[0].toStr());
            break;
         case 1:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( runSelf )");
            *rval = pthis1->runSelf(ctx);
            break;
         case 2:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( runSelfElevated )");
            pthis1->runSelfElevated(ctx,argv[0].toStr());
            break;
         case 3:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( runSimple )");
            pthis1->runSimple(argv[0].toStr());
            break;
         case 4:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( valid )");
            *rval = pthis1->valid();
            break;
         case 5:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( pid )");
            *rval = pthis1->pid();
            break;
         case 6:
            if (argc != 2)
               return o3_new(cEx)("Invalid argument count. ( send )");
            pthis1->send(argv[0].toStr(),argv[1].toInt32());
            break;
         case 7:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( kill )");
            pthis1->kill();
            break;
         case 8:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( exitCode )");
            *rval = pthis1->exitCode();
            break;
      }
      return ex;
}

siEx cProcess1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cProcess1* pthis1 = (cProcess1*) pthis;

      switch(index) {
         case 0:
            if (argc > 2)
               return o3_new(cEx)("Invalid argument count. ( process )");
            *rval = pthis1->process(ctx,argc > 0 ? argv[0].toStr() : 0,argc > 1 ? argv[1].toInt32() : 0);
            break;
      }
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
