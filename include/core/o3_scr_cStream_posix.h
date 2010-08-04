#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cStream::select()
{
   return clsTraits();
}

Trait* cStream::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cStream",            0,                    0,              0,      cStreamBase::clsTraits()  },
         {      0,      Trait::TYPE_END,        "cStream",            0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cStream::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cStream",            0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_END,        "cStream",            0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cStream::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      return ex;
}

siEx cStream::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
