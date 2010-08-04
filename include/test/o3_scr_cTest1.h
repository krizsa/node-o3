#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cTest1::select()
{
   return clsTraits();
}

Trait* cTest1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cTest1",             0,                    0,              0,      cScr::clsTraits()  },
         {      0,      Trait::TYPE_END,        "cTest1",             0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cTest1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cTest1",             0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_FUN,        "cO3",                "testBuf",            extInvoke,      0,      0                  },
         {      1,      Trait::TYPE_FUN,        "cO3",                "testVec",            extInvoke,      1,      0                  },
         {      2,      Trait::TYPE_FUN,        "cO3",                "testStr",            extInvoke,      2,      0                  },
         {      3,      Trait::TYPE_FUN,        "cO3",                "testWStr",           extInvoke,      3,      0                  },
         {      4,      Trait::TYPE_FUN,        "cO3",                "testVar",            extInvoke,      4,      0                  },
         {      5,      Trait::TYPE_FUN,        "cO3",                "testList",           extInvoke,      5,      0                  },
         {      6,      Trait::TYPE_FUN,        "cO3",                "testMap",            extInvoke,      6,      0                  },
         {      0,      Trait::TYPE_END,        "cTest1",             0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cTest1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      return ex;
}

siEx cTest1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cTest1* pthis1 = (cTest1*) pthis;

      switch(index) {
         case 0:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( testBuf )");
            pthis1->testBuf();
            break;
         case 1:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( testVec )");
            pthis1->testVec();
            break;
         case 2:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( testStr )");
            pthis1->testStr();
            break;
         case 3:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( testWStr )");
            pthis1->testWStr();
            break;
         case 4:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( testVar )");
            pthis1->testVar();
            break;
         case 5:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( testList )");
            pthis1->testList();
            break;
         case 6:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( testMap )");
            pthis1->testMap();
            break;
      }
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
