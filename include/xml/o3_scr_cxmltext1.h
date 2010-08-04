#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cXmlText1::select()
{
   return clsTraits();
}

Trait* cXmlText1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cXmlText1",          0,                    0,              0,      cXmlCharacterData1::clsTraits()  },
         {      0,      Trait::TYPE_END,        "cXmlText1",          0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cXmlText1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cXmlText1",          0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_END,        "cXmlText1",          0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cXmlText1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      return ex;
}

siEx cXmlText1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
