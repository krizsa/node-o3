#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cStaticCtrl1::select()
{
   return clsTraits();
}

Trait* cStaticCtrl1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cStaticCtrl1",       0,                    0,              0,      cWindow1::clsTraits()  },
         {      0,      Trait::TYPE_END,        "cStaticCtrl1",       0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cStaticCtrl1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cStaticCtrl1",       0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_FUN,        "cWindow1",           "createTextbox",      extInvoke,      0,      0                  },
         {      1,      Trait::TYPE_FUN,        "cWindow1",           "createBlank",        extInvoke,      1,      0                  },
         {      2,      Trait::TYPE_FUN,        "cWindow1",           "createSeparator",    extInvoke,      2,      0                  },
         {      3,      Trait::TYPE_FUN,        "cWindow1",           "createImgbox",       extInvoke,      3,      0                  },
         {      0,      Trait::TYPE_END,        "cStaticCtrl1",       0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cStaticCtrl1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      return ex;
}

siEx cStaticCtrl1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cStaticCtrl1* pthis1 = (cStaticCtrl1*) pthis;

      switch(index) {
         case 0:
            if (argc < 5 && argc > 8)
               return o3_new(cEx)("Invalid argument count. ( createTextbox )");
            *rval = siWindow(pthis1->createTextbox(pthis,argv[0].toStr(),argv[1].toInt32(),argv[2].toInt32(),argv[3].toInt32(),argv[4].toInt32(),argc > 5 ? argv[5].toInt32() : 16,argc > 6 ? argv[6].toInt32() : 0,argc > 7 ? argv[7].toInt32() : -1));
            break;
         case 1:
            if (argc < 4 && argc > 5)
               return o3_new(cEx)("Invalid argument count. ( createBlank )");
            *rval = siWindow(pthis1->createBlank(pthis,argv[0].toInt32(),argv[1].toInt32(),argv[2].toInt32(),argv[3].toInt32(),argc > 4 ? argv[4].toInt32() : 0));
            break;
         case 2:
            if (argc != 3)
               return o3_new(cEx)("Invalid argument count. ( createSeparator )");
            *rval = siWindow(pthis1->createSeparator(pthis,argv[0].toInt32(),argv[1].toInt32(),argv[2].toInt32()));
            break;
         case 3:
            if (argc != 5)
               return o3_new(cEx)("Invalid argument count. ( createImgbox )");
            *rval = siWindow(pthis1->createImgbox(pthis,argv[0].toStr(),argv[1].toInt32(),argv[2].toInt32(),argv[3].toInt32(),argv[4].toInt32()));
            break;
      }
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
