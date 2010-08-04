#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cXmlCDATASection1::select()
{
   return clsTraits();
}

Trait* cXmlCDATASection1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cXmlCDATASection1",  0,                    0,              0,      cXmlText1::clsTraits()  },
         {      0,      Trait::TYPE_END,        "cXmlCDATASection1",  0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cXmlCDATASection1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cXmlCDATASection1",  0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_END,        "cXmlCDATASection1",  0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cXmlCDATASection1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      return ex;
}

siEx cXmlCDATASection1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
