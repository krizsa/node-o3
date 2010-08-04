#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cHttp1::select()
{
   return clsTraits();
}

Trait* cHttp1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cHttp1",             0,                    0,              0,      cScr::clsTraits()  },
         {      0,      Trait::TYPE_GET,        "cHttp1",             "READY_STATE_UNINITIALIZED",clsInvoke,      0,      0                  },
         {      1,      Trait::TYPE_GET,        "cHttp1",             "READY_STATE_LOADING",clsInvoke,      1,      0                  },
         {      2,      Trait::TYPE_GET,        "cHttp1",             "READY_STATE_LOADED", clsInvoke,      2,      0                  },
         {      3,      Trait::TYPE_GET,        "cHttp1",             "READY_STATE_INTERACTIVE",clsInvoke,      3,      0                  },
         {      4,      Trait::TYPE_GET,        "cHttp1",             "READY_STATE_COMPLETED",clsInvoke,      4,      0                  },
         {      5,      Trait::TYPE_GET,        "cHttp1",             "readyState",         clsInvoke,      5,      0                  },
         {      6,      Trait::TYPE_FUN,        "cHttp1",             "open",               clsInvoke,      6,      0                  },
         {      7,      Trait::TYPE_FUN,        "cHttp1",             "setRequestHeader",   clsInvoke,      7,      0                  },
         {      8,      Trait::TYPE_FUN,        "cHttp1",             "send",               clsInvoke,      8,      0                  },
         {      8,      Trait::TYPE_FUN,        "cHttp1",             "send",               clsInvoke,      9,      0                  },
         {      9,      Trait::TYPE_GET,        "cHttp1",             "statusText",         clsInvoke,      10,     0                  },
         {      10,     Trait::TYPE_GET,        "cHttp1",             "statusCode",         clsInvoke,      11,     0                  },
         {      11,     Trait::TYPE_FUN,        "cHttp1",             "getAllResponseHeaders",clsInvoke,      12,     0                  },
         {      12,     Trait::TYPE_FUN,        "cHttp1",             "getResponseHeader",  clsInvoke,      13,     0                  },
         {      13,     Trait::TYPE_GET,        "cHttp1",             "bytesTotal",         clsInvoke,      14,     0                  },
         {      14,     Trait::TYPE_GET,        "cHttp1",             "bytesReceived",      clsInvoke,      15,     0                  },
         {      15,     Trait::TYPE_GET,        "cHttp1",             "responseBlob",       clsInvoke,      16,     0                  },
         {      16,     Trait::TYPE_GET,        "cHttp1",             "responseText",       clsInvoke,      17,     0                  },
         {      17,     Trait::TYPE_FUN,        "cHttp1",             "abort",              clsInvoke,      18,     0                  },
         {      18,     Trait::TYPE_GET,        "cHttp1",             "onreadystatechange", clsInvoke,      19,     0                  },
         {      18,     Trait::TYPE_SET,        "cHttp1",             "onreadystatechange", clsInvoke,      20,     0                  },
         {      19,     Trait::TYPE_GET,        "cHttp1",             "onprogress",         clsInvoke,      21,     0                  },
         {      19,     Trait::TYPE_SET,        "cHttp1",             "onprogress",         clsInvoke,      22,     0                  },
         {      20,     Trait::TYPE_FUN,        "cHttp1",             "responseOpen",       clsInvoke,      23,     0                  },
         {      0,      Trait::TYPE_END,        "cHttp1",             0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cHttp1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cHttp1",             0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_FUN,        "cO3",                "http",               extInvoke,      0,      0                  },
         {      0,      Trait::TYPE_END,        "cHttp1",             0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cHttp1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cHttp1* pthis1 = (cHttp1*) pthis;

      switch(index) {
         case 0:
            *rval = 0;
            break;
         case 1:
            *rval = 1;
            break;
         case 2:
            *rval = 2;
            break;
         case 3:
            *rval = 3;
            break;
         case 4:
            *rval = 4;
            break;
         case 5:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( readyState )");
            *rval = pthis1->readyState();
            break;
         case 6:
            if (argc < 2 && argc > 3)
               return o3_new(cEx)("Invalid argument count. ( open )");
            pthis1->open(argv[0].toStr(),argv[1].toStr(),argc > 2 ? argv[2].toBool() : true);
            break;
         case 7:
            if (argc != 2)
               return o3_new(cEx)("Invalid argument count. ( setRequestHeader )");
            pthis1->setRequestHeader(argv[0].toStr(),argv[1].toStr());
            break;
         case 8:
            if (argc==1) {
               Var::Type type0 = argv[0].type();
               if (siBuf sibuf = siBuf(argv[0].toScr())) {
                  pthis1->send(ctx,Buf(siBuf(argv[0].toScr())));
                  return ex;
               }
               else if(Var::TYPE_VOID <= type0 && Var::TYPE_SCR >= type0) {
                  pthis1->send(ctx,argv[0].toStr());
               }
               else{
                  return o3_new(cEx)("Invalid argument type.");
               }
            }
            else{
               return o3_new(cEx)("Invalid argument count.");
            }
            break;
         case 10:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( statusText )");
            *rval = pthis1->statusText();
            break;
         case 11:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( statusCode )");
            *rval = pthis1->statusCode();
            break;
         case 12:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( getAllResponseHeaders )");
            *rval = pthis1->getAllResponseHeaders();
            break;
         case 13:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( getResponseHeader )");
            *rval = pthis1->getResponseHeader(argv[0].toStr());
            break;
         case 14:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( bytesTotal )");
            *rval = pthis1->bytesTotal();
            break;
         case 15:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( bytesReceived )");
            *rval = pthis1->bytesReceived();
            break;
         case 16:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( responseBlob )");
            *rval = o3_new(cScrBuf)(pthis1->responseBlob());
            break;
         case 17:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( responseText )");
            *rval = pthis1->responseText();
            break;
         case 18:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( abort )");
            pthis1->abort();
            break;
         case 19:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( onreadystatechange )");
            *rval = pthis1->onreadystatechange();
            break;
         case 20:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( setOnreadystatechange )");
            *rval = pthis1->setOnreadystatechange(ctx,argv[0].toScr());
            break;
         case 21:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( onprogress )");
            *rval = pthis1->onprogress();
            break;
         case 22:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( setOnprogress )");
            *rval = pthis1->setOnprogress(ctx,argv[0].toScr());
            break;
         case 23:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( responseOpen )");
            *rval = siFs(pthis1->responseOpen(siFs (argv[0].toScr())));
            break;
      }
      return ex;
}

siEx cHttp1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cHttp1* pthis1 = (cHttp1*) pthis;

      switch(index) {
         case 0:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( http )");
            *rval = pthis1->http();
            break;
      }
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
