#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cWindow1::select()
{
   return clsTraits();
}

Trait* cWindow1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cWindow1",           0,                    0,              0,      cWindow1Base::clsTraits()  },
         {      0,      Trait::TYPE_GET,        "cWindow1",           "BOLD",               clsInvoke,      0,      0                  },
         {      1,      Trait::TYPE_GET,        "cWindow1",           "ITALIC",             clsInvoke,      1,      0                  },
         {      2,      Trait::TYPE_GET,        "cWindow1",           "UNDERLINE",          clsInvoke,      2,      0                  },
         {      3,      Trait::TYPE_GET,        "cWindow1",           "STRIKEOUT",          clsInvoke,      3,      0                  },
         {      4,      Trait::TYPE_FUN,        "cWindow1",           "createWindow",       clsInvoke,      4,      0                  },
         {      5,      Trait::TYPE_GET,        "cWindow1",           "clientX",            clsInvoke,      5,      0                  },
         {      6,      Trait::TYPE_GET,        "cWindow1",           "clientY",            clsInvoke,      6,      0                  },
         {      7,      Trait::TYPE_FUN,        "cWindow1",           "minimize",           clsInvoke,      7,      0                  },
         {      8,      Trait::TYPE_FUN,        "cWindow1",           "maximize",           clsInvoke,      8,      0                  },
         {      9,      Trait::TYPE_FUN,        "cWindow1",           "restore",            clsInvoke,      9,      0                  },
         {      10,     Trait::TYPE_FUN,        "cWindow1",           "close",              clsInvoke,      10,     0                  },
         {      11,     Trait::TYPE_SET,        "cWindow1",           "visible",            clsInvoke,      11,     0                  },
         {      12,     Trait::TYPE_GET,        "cWindow1",           "showButtons",        clsInvoke,      12,     0                  },
         {      12,     Trait::TYPE_SET,        "cWindow1",           "showButtons",        clsInvoke,      13,     0                  },
         {      13,     Trait::TYPE_GET,        "cWindow1",           "caption",            clsInvoke,      14,     0                  },
         {      13,     Trait::TYPE_SET,        "cWindow1",           "caption",            clsInvoke,      15,     0                  },
         {      14,     Trait::TYPE_SET,        "cWindow1",           "icon",               clsInvoke,      16,     0                  },
         {      15,     Trait::TYPE_FUN,        "cWindow1",           "useIcon",            clsInvoke,      17,     0                  },
         {      16,     Trait::TYPE_FUN,        "cWindow1",           "focus",              clsInvoke,      18,     0                  },
         {      17,     Trait::TYPE_FUN,        "cWindow1",           "destroy",            clsInvoke,      19,     0                  },
         {      18,     Trait::TYPE_SET,        "cWindow1",           "onclose",            clsInvoke,      20,     0                  },
         {      19,     Trait::TYPE_SET,        "cWindow1",           "onend",              clsInvoke,      21,     0                  },
         {      0,      Trait::TYPE_END,        "cWindow1",           0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cWindow1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cWindow1",           0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_GET,        "cO3",                "window",             extInvoke,      0,      0                  },
         {      1,      Trait::TYPE_FUN,        "cO3",                "createWindow",       extInvoke,      1,      0                  },
         {      2,      Trait::TYPE_FUN,        "cO3",                "sendKeyDown",        extInvoke,      2,      0                  },
         {      3,      Trait::TYPE_FUN,        "cO3",                "sendKeyUp",          extInvoke,      3,      0                  },
         {      4,      Trait::TYPE_FUN,        "cO3",                "sendKey",            extInvoke,      4,      0                  },
         {      5,      Trait::TYPE_FUN,        "cO3",                "sendAsKeyEvents",    extInvoke,      5,      0                  },
         {      6,      Trait::TYPE_FUN,        "cO3",                "mouseTo",            extInvoke,      6,      0                  },
         {      7,      Trait::TYPE_FUN,        "cO3",                "mouseLeftClick",     extInvoke,      7,      0                  },
         {      8,      Trait::TYPE_FUN,        "cO3",                "mouseRightClick",    extInvoke,      8,      0                  },
         {      9,      Trait::TYPE_FUN,        "cO3",                "mouseLeftDown",      extInvoke,      9,      0                  },
         {      10,     Trait::TYPE_FUN,        "cO3",                "mouseLeftUp",        extInvoke,      10,     0                  },
         {      11,     Trait::TYPE_FUN,        "cO3",                "mouseRightDown",     extInvoke,      11,     0                  },
         {      12,     Trait::TYPE_FUN,        "cO3",                "mouseRightUp",       extInvoke,      12,     0                  },
         {      13,     Trait::TYPE_FUN,        "cO3",                "mouseWheel",         extInvoke,      13,     0                  },
         {      14,     Trait::TYPE_FUN,        "cO3",                "alertBox",           extInvoke,      14,     0                  },
         {      0,      Trait::TYPE_END,        "cWindow1",           0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cWindow1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cWindow1* pthis1 = (cWindow1*) pthis;

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
            if (argc < 5 && argc > 6)
               return o3_new(cEx)("Invalid argument count. ( createWindow )");
            *rval = siWindow(pthis1->createWindow(argv[0].toStr(),argv[1].toInt32(),argv[2].toInt32(),argv[3].toInt32(),argv[4].toInt32(),argc > 5 ? argv[5].toInt32() : 0));
            break;
         case 5:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( clientX )");
            *rval = pthis1->clientX();
            break;
         case 6:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( clientY )");
            *rval = pthis1->clientY();
            break;
         case 7:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( minimize )");
            pthis1->minimize();
            break;
         case 8:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( maximize )");
            pthis1->maximize();
            break;
         case 9:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( restore )");
            pthis1->restore();
            break;
         case 10:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( close )");
            pthis1->close();
            break;
         case 11:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( setVisible )");
            *rval = pthis1->setVisible(argv[0].toBool());
            break;
         case 12:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( showButtons )");
            *rval = pthis1->showButtons();
            break;
         case 13:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( setShowButtons )");
            *rval = pthis1->setShowButtons(argv[0].toBool());
            break;
         case 14:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( caption )");
            *rval = pthis1->caption();
            break;
         case 15:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( setCaption )");
            *rval = pthis1->setCaption(argv[0].toStr());
            break;
         case 16:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( setIcon )");
            pthis1->setIcon(argv[0].toStr());
            break;
         case 17:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( useIcon )");
            pthis1->useIcon(Buf(siBuf(argv[0].toScr())));
            break;
         case 18:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( focus )");
            *rval = pthis1->focus();
            break;
         case 19:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( destroy )");
            pthis1->destroy();
            break;
         case 20:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( setOnclose )");
            *rval = pthis1->setOnclose(ctx,argv[0].toScr());
            break;
         case 21:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( setOnend )");
            *rval = pthis1->setOnend(ctx,argv[0].toScr());
            break;
      }
      return ex;
}

siEx cWindow1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cWindow1* pthis1 = (cWindow1*) pthis;

      switch(index) {
         case 0:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( window )");
            *rval = siWindow(pthis1->window(ctx));
            break;
         case 1:
            if (argc < 5 && argc > 6)
               return o3_new(cEx)("Invalid argument count. ( createWindow )");
            *rval = siWindow(pthis1->createWindow(pthis,argv[0].toStr(),argv[1].toInt32(),argv[2].toInt32(),argv[3].toInt32(),argv[4].toInt32(),argc > 5 ? argv[5].toInt32() : 0));
            break;
         case 2:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( sendKeyDown )");
            pthis1->sendKeyDown(argv[0].toInt32());
            break;
         case 3:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( sendKeyUp )");
            pthis1->sendKeyUp(argv[0].toInt32());
            break;
         case 4:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( sendKey )");
            pthis1->sendKey(argv[0].toInt32());
            break;
         case 5:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( sendAsKeyEvents )");
            pthis1->sendAsKeyEvents(argv[0].toStr());
            break;
         case 6:
            if (argc < 2 && argc > 3)
               return o3_new(cEx)("Invalid argument count. ( mouseTo )");
            pthis1->mouseTo(argv[0].toInt32(),argv[1].toInt32(),siWindow (argc > 2 ? argv[2].toScr() : 0));
            break;
         case 7:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( mouseLeftClick )");
            pthis1->mouseLeftClick();
            break;
         case 8:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( mouseRightClick )");
            pthis1->mouseRightClick();
            break;
         case 9:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( mouseLeftDown )");
            pthis1->mouseLeftDown();
            break;
         case 10:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( mouseLeftUp )");
            pthis1->mouseLeftUp();
            break;
         case 11:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( mouseRightDown )");
            pthis1->mouseRightDown();
            break;
         case 12:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( mouseRightUp )");
            pthis1->mouseRightUp();
            break;
         case 13:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( mouseWheel )");
            pthis1->mouseWheel(argv[0].toInt32());
            break;
         case 14:
            if (argc < 2 && argc > 3)
               return o3_new(cEx)("Invalid argument count. ( alertBox )");
            *rval = pthis1->alertBox(argv[0].toStr(),argv[1].toStr(),argc > 2 ? argv[2].toStr() : 0,&ex);
            break;
      }
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
