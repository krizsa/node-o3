#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cWindow1Base::select()
{
   return clsTraits();
}

Trait* cWindow1Base::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cWindow1Base",       0,                    0,              0,      cScr::clsTraits()  },
         {      0,      Trait::TYPE_GET,        "cWindow1Base",       "x",                  clsInvoke,      0,      0                  },
         {      0,      Trait::TYPE_SET,        "cWindow1Base",       "x",                  clsInvoke,      1,      0                  },
         {      1,      Trait::TYPE_GET,        "cWindow1Base",       "y",                  clsInvoke,      2,      0                  },
         {      1,      Trait::TYPE_SET,        "cWindow1Base",       "y",                  clsInvoke,      3,      0                  },
         {      2,      Trait::TYPE_GET,        "cWindow1Base",       "width",              clsInvoke,      4,      0                  },
         {      2,      Trait::TYPE_SET,        "cWindow1Base",       "width",              clsInvoke,      5,      0                  },
         {      3,      Trait::TYPE_GET,        "cWindow1Base",       "height",             clsInvoke,      6,      0                  },
         {      3,      Trait::TYPE_SET,        "cWindow1Base",       "height",             clsInvoke,      7,      0                  },
         {      0,      Trait::TYPE_END,        "cWindow1Base",       0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cWindow1Base::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cWindow1Base",       0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_END,        "cWindow1Base",       0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cWindow1Base::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cWindow1Base* pthis1 = (cWindow1Base*) pthis;

      switch(index) {
         case 0:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( x )");
            *rval = pthis1->x();
            break;
         case 1:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( setX )");
            *rval = pthis1->setX(argv[0].toInt32());
            break;
         case 2:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( y )");
            *rval = pthis1->y();
            break;
         case 3:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( setY )");
            *rval = pthis1->setY(argv[0].toInt32());
            break;
         case 4:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( width )");
            *rval = pthis1->width();
            break;
         case 5:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( setWidth )");
            *rval = pthis1->setWidth(argv[0].toInt32());
            break;
         case 6:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( height )");
            *rval = pthis1->height();
            break;
         case 7:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( setHeight )");
            *rval = pthis1->setHeight(argv[0].toInt32());
            break;
      }
      return ex;
}

siEx cWindow1Base::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
