#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cStreamBase::select()
{
   return clsTraits();
}

Trait* cStreamBase::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cStreamBase",        0,                    0,              0,      cScr::clsTraits()  },
         {      0,      Trait::TYPE_GET,        "cStreamBase",        "eof",                clsInvoke,      0,      0                  },
         {      1,      Trait::TYPE_GET,        "cStreamBase",        "error",              clsInvoke,      1,      0                  },
         {      2,      Trait::TYPE_GET,        "cStreamBase",        "pos",                clsInvoke,      2,      0                  },
         {      2,      Trait::TYPE_SET,        "cStreamBase",        "pos",                clsInvoke,      3,      0                  },
         {      3,      Trait::TYPE_FUN,        "cStreamBase",        "readBlob",           clsInvoke,      4,      0                  },
         {      4,      Trait::TYPE_FUN,        "cStreamBase",        "read",               clsInvoke,      5,      0                  },
         {      5,      Trait::TYPE_FUN,        "cStreamBase",        "write",              clsInvoke,      6,      0                  },
         {      5,      Trait::TYPE_FUN,        "cStreamBase",        "write",              clsInvoke,      7,      0                  },
         {      6,      Trait::TYPE_FUN,        "cStreamBase",        "flush",              clsInvoke,      8,      0                  },
         {      7,      Trait::TYPE_FUN,        "cStreamBase",        "close",              clsInvoke,      9,      0                  },
         {      8,      Trait::TYPE_FUN,        "cStreamBase",        "print",              clsInvoke,      10,     0                  },
         {      0,      Trait::TYPE_END,        "cStreamBase",        0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cStreamBase::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cStreamBase",        0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_END,        "cStreamBase",        0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cStreamBase::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cStreamBase* pthis1 = (cStreamBase*) pthis;

      switch(index) {
         case 0:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( eof )");
            *rval = pthis1->eof();
            break;
         case 1:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( error )");
            *rval = pthis1->error();
            break;
         case 2:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( pos )");
            *rval = pthis1->pos();
            break;
         case 3:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( setPos )");
            *rval = pthis1->setPos(argv[0].toInt32());
            break;
         case 4:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( readBlob )");
            *rval = o3_new(cScrBuf)(pthis1->readBlob(argv[0].toInt32()));
            break;
         case 5:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( read )");
            *rval = pthis1->read(argv[0].toInt32());
            break;
         case 6:
            if (argc==1) {
               Var::Type type0 = argv[0].type();
               if (siBuf sibuf = siBuf(argv[0].toScr())) {
                  *rval = pthis1->write(Buf(siBuf(argv[0].toScr())));
                  return ex;
               }
               else if(Var::TYPE_VOID <= type0 && Var::TYPE_SCR >= type0) {
                  *rval = pthis1->write(argv[0].toStr());
               }
               else{
                  return o3_new(cEx)("Invalid argument type.");
               }
            }
            else{
               return o3_new(cEx)("Invalid argument count.");
            }
            break;
         case 8:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( flush )");
            *rval = pthis1->flush();
            break;
         case 9:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( close )");
            *rval = pthis1->close();
            break;
         case 10:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( print )");
            pthis1->print(argv[0].toStr());
            break;
      }
      return ex;
}

siEx cStreamBase::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
