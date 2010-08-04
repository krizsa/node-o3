#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cProcess1Base::select()
{
   return clsTraits();
}

Trait* cProcess1Base::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cProcess1Base",      0,                    0,              0,      cScr::clsTraits()  },
         {      0,      Trait::TYPE_GET,        "cProcess1Base",      "stdIn",              clsInvoke,      0,      0                  },
         {      0,      Trait::TYPE_SET,        "cProcess1Base",      "stdIn",              clsInvoke,      1,      0                  },
         {      1,      Trait::TYPE_GET,        "cProcess1Base",      "stdOut",             clsInvoke,      2,      0                  },
         {      1,      Trait::TYPE_SET,        "cProcess1Base",      "stdOut",             clsInvoke,      3,      0                  },
         {      2,      Trait::TYPE_GET,        "cProcess1Base",      "stdErr",             clsInvoke,      4,      0                  },
         {      2,      Trait::TYPE_SET,        "cProcess1Base",      "stdErr",             clsInvoke,      5,      0                  },
         {      3,      Trait::TYPE_GET,        "cProcess1Base",      "onterminate",        clsInvoke,      6,      0                  },
         {      3,      Trait::TYPE_SET,        "cProcess1Base",      "onterminate",        clsInvoke,      7,      0                  },
         {      4,      Trait::TYPE_FUN,        "cProcess1Base",      "exec",               clsInvoke,      8,      0                  },
         {      5,      Trait::TYPE_GET,        "cProcess1Base",      "exitCode",           clsInvoke,      9,      0                  },
         {      0,      Trait::TYPE_END,        "cProcess1Base",      0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cProcess1Base::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cProcess1Base",      0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_END,        "cProcess1Base",      0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cProcess1Base::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cProcess1Base* pthis1 = (cProcess1Base*) pthis;

      switch(index) {
         case 0:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( stdIn )");
            *rval = siStream(pthis1->stdIn());
            break;
         case 1:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( setStdIn )");
            *rval = siStream(pthis1->setStdIn(siStream (argv[0].toScr())));
            break;
         case 2:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( stdOut )");
            *rval = siStream(pthis1->stdOut());
            break;
         case 3:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( setStdOut )");
            *rval = siStream(pthis1->setStdOut(siStream (argv[0].toScr())));
            break;
         case 4:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( stdErr )");
            *rval = siStream(pthis1->stdErr());
            break;
         case 5:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( setStdErr )");
            *rval = siStream(pthis1->setStdErr(siStream (argv[0].toScr())));
            break;
         case 6:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( onterminate )");
            *rval = pthis1->onterminate();
            break;
         case 7:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( setOnterminate )");
            *rval = pthis1->setOnterminate(ctx,argv[0].toScr());
            break;
         case 8:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( exec )");
            pthis1->exec(ctx,argv[0].toStr());
            break;
         case 9:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( exitCode )");
            *rval = pthis1->exitCode();
            break;
      }
      return ex;
}

siEx cProcess1Base::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
