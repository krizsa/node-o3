#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cScreen1Base::select()
{
   return clsTraits();
}

Trait* cScreen1Base::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cScreen1Base",       0,                    0,              0,      cScr::clsTraits()  },
         {      0,      Trait::TYPE_GET,        "cScreen1Base",       "width",              clsInvoke,      0,      0                  },
         {      1,      Trait::TYPE_GET,        "cScreen1Base",       "height",             clsInvoke,      1,      0                  },
         {      0,      Trait::TYPE_END,        "cScreen1Base",       0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cScreen1Base::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cScreen1Base",       0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_END,        "cScreen1Base",       0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cScreen1Base::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cScreen1Base* pthis1 = (cScreen1Base*) pthis;

      switch(index) {
         case 0:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( width )");
            *rval = pthis1->width();
            break;
         case 1:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( height )");
            *rval = pthis1->height();
            break;
      }
      return ex;
}

siEx cScreen1Base::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
