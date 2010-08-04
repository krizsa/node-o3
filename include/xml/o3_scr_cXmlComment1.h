#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cXmlComment1::select()
{
   return clsTraits();
}

Trait* cXmlComment1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cXmlComment1",       0,                    0,              0,      cXmlCharacterData1::clsTraits()  },
         {      0,      Trait::TYPE_END,        "cXmlComment1",       0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cXmlComment1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cXmlComment1",       0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_END,        "cXmlComment1",       0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cXmlComment1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      return ex;
}

siEx cXmlComment1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
