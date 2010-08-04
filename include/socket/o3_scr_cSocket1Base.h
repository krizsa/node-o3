#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cSocket1Base::select()
{
   return clsTraits();
}

Trait* cSocket1Base::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cSocket1Base",       0,                    0,              0,      cScr::clsTraits()  },
         {      0,      Trait::TYPE_GET,        "cSocket1Base",       "STATE_ERROR",        clsInvoke,      0,      0                  },
         {      1,      Trait::TYPE_GET,        "cSocket1Base",       "STATE_CLOSED",       clsInvoke,      1,      0                  },
         {      2,      Trait::TYPE_GET,        "cSocket1Base",       "STATE_CREATED",      clsInvoke,      2,      0                  },
         {      3,      Trait::TYPE_GET,        "cSocket1Base",       "STATE_CONNECTED",    clsInvoke,      3,      0                  },
         {      4,      Trait::TYPE_GET,        "cSocket1Base",       "STATE_CONNECTING",   clsInvoke,      4,      0                  },
         {      5,      Trait::TYPE_GET,        "cSocket1Base",       "STATE_ACCEPTING",    clsInvoke,      5,      0                  },
         {      6,      Trait::TYPE_GET,        "cSocket1Base",       "STATE_RECEIVING",    clsInvoke,      6,      0                  },
         {      7,      Trait::TYPE_GET,        "cSocket1Base",       "STATE_SENDING",      clsInvoke,      7,      0                  },
         {      8,      Trait::TYPE_GET,        "cSocket1Base",       "TYPE_INVALID",       clsInvoke,      8,      0                  },
         {      9,      Trait::TYPE_GET,        "cSocket1Base",       "TYPE_UDP",           clsInvoke,      9,      0                  },
         {      10,     Trait::TYPE_GET,        "cSocket1Base",       "TYPE_TCP",           clsInvoke,      10,     0                  },
         {      11,     Trait::TYPE_GET,        "cSocket1Base",       "error",              clsInvoke,      11,     0                  },
         {      12,     Trait::TYPE_GET,        "cSocket1Base",       "isClosed",           clsInvoke,      12,     0                  },
         {      13,     Trait::TYPE_GET,        "cSocket1Base",       "isCreated",          clsInvoke,      13,     0                  },
         {      14,     Trait::TYPE_GET,        "cSocket1Base",       "isConnected",        clsInvoke,      14,     0                  },
         {      15,     Trait::TYPE_GET,        "cSocket1Base",       "isConnecting",       clsInvoke,      15,     0                  },
         {      16,     Trait::TYPE_GET,        "cSocket1Base",       "isAccepting",        clsInvoke,      16,     0                  },
         {      17,     Trait::TYPE_GET,        "cSocket1Base",       "isReceiveing",       clsInvoke,      17,     0                  },
         {      18,     Trait::TYPE_GET,        "cSocket1Base",       "isSending",          clsInvoke,      18,     0                  },
         {      19,     Trait::TYPE_FUN,        "cSocket1Base",       "bind",               clsInvoke,      19,     0                  },
         {      19,     Trait::TYPE_FUN,        "cSocket1Base",       "bind",               clsInvoke,      20,     0                  },
         {      19,     Trait::TYPE_FUN,        "cSocket1Base",       "bind",               clsInvoke,      21,     0                  },
         {      20,     Trait::TYPE_FUN,        "cSocket1Base",       "connect",            clsInvoke,      22,     0                  },
         {      20,     Trait::TYPE_FUN,        "cSocket1Base",       "connect",            clsInvoke,      23,     0                  },
         {      21,     Trait::TYPE_FUN,        "cSocket1Base",       "accept",             clsInvoke,      24,     0                  },
         {      22,     Trait::TYPE_FUN,        "cSocket1Base",       "receive",            clsInvoke,      25,     0                  },
         {      23,     Trait::TYPE_FUN,        "cSocket1Base",       "send",               clsInvoke,      26,     0                  },
         {      23,     Trait::TYPE_FUN,        "cSocket1Base",       "send",               clsInvoke,      27,     0                  },
         {      24,     Trait::TYPE_FUN,        "cSocket1Base",       "sendTo",             clsInvoke,      28,     0                  },
         {      24,     Trait::TYPE_FUN,        "cSocket1Base",       "sendTo",             clsInvoke,      29,     0                  },
         {      25,     Trait::TYPE_GET,        "cSocket1Base",       "receivedBuf",        clsInvoke,      30,     0                  },
         {      26,     Trait::TYPE_GET,        "cSocket1Base",       "receivedText",       clsInvoke,      31,     0                  },
         {      27,     Trait::TYPE_FUN,        "cSocket1Base",       "clearBuf",           clsInvoke,      32,     0                  },
         {      28,     Trait::TYPE_GET,        "cSocket1Base",       "onaccept",           clsInvoke,      33,     0                  },
         {      28,     Trait::TYPE_SET,        "cSocket1Base",       "onaccept",           clsInvoke,      34,     0                  },
         {      29,     Trait::TYPE_GET,        "cSocket1Base",       "onconnect",          clsInvoke,      35,     0                  },
         {      29,     Trait::TYPE_SET,        "cSocket1Base",       "onconnect",          clsInvoke,      36,     0                  },
         {      30,     Trait::TYPE_GET,        "cSocket1Base",       "onreceive",          clsInvoke,      37,     0                  },
         {      30,     Trait::TYPE_SET,        "cSocket1Base",       "onreceive",          clsInvoke,      38,     0                  },
         {      31,     Trait::TYPE_GET,        "cSocket1Base",       "onsend",             clsInvoke,      39,     0                  },
         {      31,     Trait::TYPE_SET,        "cSocket1Base",       "onsend",             clsInvoke,      40,     0                  },
         {      32,     Trait::TYPE_FUN,        "cSocket1Base",       "close",              clsInvoke,      41,     0                  },
         {      33,     Trait::TYPE_GET,        "cSocket1Base",       "packetSize",         clsInvoke,      42,     0                  },
         {      33,     Trait::TYPE_SET,        "cSocket1Base",       "packetSize",         clsInvoke,      43,     0                  },
         {      34,     Trait::TYPE_GET,        "cSocket1Base",       "bytesSent",          clsInvoke,      44,     0                  },
         {      35,     Trait::TYPE_GET,        "cSocket1Base",       "bytesReceived",      clsInvoke,      45,     0                  },
         {      36,     Trait::TYPE_GET,        "cSocket1Base",       "type",               clsInvoke,      46,     0                  },
         {      37,     Trait::TYPE_GET,        "cSocket1Base",       "srcAddress",         clsInvoke,      47,     0                  },
         {      0,      Trait::TYPE_END,        "cSocket1Base",       0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cSocket1Base::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cSocket1Base",       0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_END,        "cSocket1Base",       0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cSocket1Base::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cSocket1Base* pthis1 = (cSocket1Base*) pthis;

      switch(index) {
         case 0:
            *rval = 1;
            break;
         case 1:
            *rval = 2;
            break;
         case 2:
            *rval = 4;
            break;
         case 3:
            *rval = 8;
            break;
         case 4:
            *rval = 16;
            break;
         case 5:
            *rval = 32;
            break;
         case 6:
            *rval = 64;
            break;
         case 7:
            *rval = 128;
            break;
         case 8:
            *rval = 0;
            break;
         case 9:
            *rval = 1;
            break;
         case 10:
            *rval = 2;
            break;
         case 11:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( error )");
            *rval = pthis1->error();
            break;
         case 12:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( isClosed )");
            *rval = pthis1->isClosed();
            break;
         case 13:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( isCreated )");
            *rval = pthis1->isCreated();
            break;
         case 14:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( isConnected )");
            *rval = pthis1->isConnected();
            break;
         case 15:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( isConnecting )");
            *rval = pthis1->isConnecting();
            break;
         case 16:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( isAccepting )");
            *rval = pthis1->isAccepting();
            break;
         case 17:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( isReceiveing )");
            *rval = pthis1->isReceiveing();
            break;
         case 18:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( isSending )");
            *rval = pthis1->isSending();
            break;
         case 19:
            if (argc==1) {
               Var::Type type0 = argv[0].type();
               if (Var::TYPE_VOID <= type0 && Var::TYPE_INT32 >= type0) {
                  *rval = pthis1->bind(argv[0].toInt32());
               }
               else if(Var::TYPE_INT64 <= type0 && Var::TYPE_SCR >= type0) {
                  *rval = pthis1->bind(argv[0].toStr());
               }
               else{
                  return o3_new(cEx)("Invalid argument type.");
               }
            }
            else if(argc==2) {
               *rval = pthis1->bind(argv[0].toStr(),argv[1].toInt32());
            }
            else{
               return o3_new(cEx)("Invalid argument count.");
            }
            break;
         case 22:
            if (argc==1) {
               *rval = pthis1->connect(argv[0].toStr());
            }
            else if(argc==2) {
               *rval = pthis1->connect(argv[0].toStr(),argv[1].toInt32());
            }
            else{
               return o3_new(cEx)("Invalid argument count.");
            }
            break;
         case 24:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( accept )");
            *rval = pthis1->accept();
            break;
         case 25:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( receive )");
            *rval = pthis1->receive();
            break;
         case 26:
            if (argc==1) {
               Var::Type type0 = argv[0].type();
               if (siBuf  sibuf  = siBuf (argv[0].toScr())) {
                  *rval = pthis1->send(siBuf (argv[0].toScr()));
                  return ex;
               }
               else if(Var::TYPE_VOID <= type0 && Var::TYPE_SCR >= type0) {
                  *rval = pthis1->send(argv[0].toStr());
               }
               else{
                  return o3_new(cEx)("Invalid argument type.");
               }
            }
            else{
               return o3_new(cEx)("Invalid argument count.");
            }
            break;
         case 28:
            if (argc==3) {
               *rval = pthis1->sendTo(siBuf (argv[0].toScr()),argv[1].toStr(),argv[2].toInt32());
            }
            else if(argc==2) {
               *rval = pthis1->sendTo(siBuf (argv[0].toScr()),argv[1].toStr());
            }
            else{
               return o3_new(cEx)("Invalid argument count.");
            }
            break;
         case 30:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( receivedBuf )");
            *rval = o3_new(cScrBuf)(pthis1->receivedBuf());
            break;
         case 31:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( receivedText )");
            *rval = pthis1->receivedText();
            break;
         case 32:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( clearBuf )");
            pthis1->clearBuf();
            break;
         case 33:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( onaccept )");
            *rval = pthis1->onaccept();
            break;
         case 34:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( setOnaccept )");
            *rval = pthis1->setOnaccept(argv[0].toScr());
            break;
         case 35:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( onconnect )");
            *rval = pthis1->onconnect();
            break;
         case 36:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( setOnconnect )");
            *rval = pthis1->setOnconnect(argv[0].toScr());
            break;
         case 37:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( onreceive )");
            *rval = pthis1->onreceive();
            break;
         case 38:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( setOnreceive )");
            *rval = pthis1->setOnreceive(argv[0].toScr());
            break;
         case 39:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( onsend )");
            *rval = pthis1->onsend();
            break;
         case 40:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( setOnsend )");
            *rval = pthis1->setOnsend(argv[0].toScr());
            break;
         case 41:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( close )");
            pthis1->close();
            break;
         case 42:
            *rval = pthis1->m_packet_size;
            break;
         case 43:
            pthis1->m_packet_size = argv[0].toInt32();
            break;
         case 44:
            *rval = pthis1->m_bytes_sent;
            break;
         case 45:
            *rval = pthis1->m_bytes_received;
            break;
         case 46:
            *rval = pthis1->m_type;
            break;
         case 47:
            *rval = pthis1->m_src_address;
            break;
      }
      return ex;
}

siEx cSocket1Base::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
