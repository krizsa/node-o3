#ifdef O3_WITH_GLUE
#pragma warning( disable : 4100)
#pragma warning( disable : 4189)
namespace o3 {


Trait* cImage1::select()
{
   return clsTraits();
}

Trait* cImage1::clsTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cImage1",            0,                    0,              0,      cScr::clsTraits()  },
         {      0,      Trait::TYPE_GET,        "cImage1",            "mode",               clsInvoke,      0,      0                  },
         {      1,      Trait::TYPE_GET,        "cImage1",            "x",                  clsInvoke,      1,      0                  },
         {      2,      Trait::TYPE_GET,        "cImage1",            "y",                  clsInvoke,      2,      0                  },
         {      3,      Trait::TYPE_GET,        "cImage1",            "width",              clsInvoke,      3,      0                  },
         {      4,      Trait::TYPE_GET,        "cImage1",            "height",             clsInvoke,      4,      0                  },
         {      5,      Trait::TYPE_FUN,        "cImage1",            "clear",              clsInvoke,      5,      0                  },
         {      6,      Trait::TYPE_FUN,        "cImage1",            "setPixel",           clsInvoke,      6,      0                  },
         {      7,      Trait::TYPE_FUN,        "cImage1",            "getPixel",           clsInvoke,      7,      0                  },
         {      8,      Trait::TYPE_SET,        "cImage1",            "src",                clsInvoke,      8,      0                  },
         {      9,      Trait::TYPE_FUN,        "cImage1",            "savePng",            clsInvoke,      9,      0                  },
         {      10,     Trait::TYPE_FUN,        "cImage1",            "rect",               clsInvoke,      10,     0                  },
         {      11,     Trait::TYPE_FUN,        "cImage1",            "line",               clsInvoke,      11,     0                  },
         {      12,     Trait::TYPE_FUN,        "cImage1",            "decodeColor",        clsInvoke,      12,     0                  },
         {      13,     Trait::TYPE_SET,        "cImage1",            "fillStyle",          clsInvoke,      13,     0                  },
         {      14,     Trait::TYPE_SET,        "cImage1",            "strokeStyle",        clsInvoke,      14,     0                  },
         {      15,     Trait::TYPE_SET,        "cImage1",            "strokeWidth",        clsInvoke,      15,     0                  },
         {      16,     Trait::TYPE_FUN,        "cImage1",            "clearRect",          clsInvoke,      16,     0                  },
         {      17,     Trait::TYPE_FUN,        "cImage1",            "fillRect",           clsInvoke,      17,     0                  },
         {      18,     Trait::TYPE_FUN,        "cImage1",            "strokeRect",         clsInvoke,      18,     0                  },
         {      19,     Trait::TYPE_FUN,        "cImage1",            "moveTo",             clsInvoke,      19,     0                  },
         {      20,     Trait::TYPE_FUN,        "cImage1",            "lineTo",             clsInvoke,      20,     0                  },
         {      21,     Trait::TYPE_FUN,        "cImage1",            "closePath",          clsInvoke,      21,     0                  },
         {      22,     Trait::TYPE_FUN,        "cImage1",            "beginPath",          clsInvoke,      22,     0                  },
         {      23,     Trait::TYPE_FUN,        "cImage1",            "fill",               clsInvoke,      23,     0                  },
         {      24,     Trait::TYPE_FUN,        "cImage1",            "stroke",             clsInvoke,      24,     0                  },
         {      25,     Trait::TYPE_FUN,        "cImage1",            "quadraticCurveTo",   clsInvoke,      25,     0                  },
         {      26,     Trait::TYPE_FUN,        "cImage1",            "bezierCurveTo",      clsInvoke,      26,     0                  },
         {      27,     Trait::TYPE_FUN,        "cImage1",            "translate",          clsInvoke,      27,     0                  },
         {      28,     Trait::TYPE_FUN,        "cImage1",            "rotate",             clsInvoke,      28,     0                  },
         {      29,     Trait::TYPE_FUN,        "cImage1",            "scale",              clsInvoke,      29,     0                  },
         {      30,     Trait::TYPE_FUN,        "cImage1",            "save",               clsInvoke,      30,     0                  },
         {      31,     Trait::TYPE_FUN,        "cImage1",            "restore",            clsInvoke,      31,     0                  },
         {      32,     Trait::TYPE_FUN,        "cImage1",            "setTransform",       clsInvoke,      32,     0                  },
         {      33,     Trait::TYPE_FUN,        "cImage1",            "transform",          clsInvoke,      33,     0                  },
         {      34,     Trait::TYPE_SET,        "cImage1",            "lineCap",            clsInvoke,      34,     0                  },
         {      35,     Trait::TYPE_SET,        "cImage1",            "lineJoin",           clsInvoke,      35,     0                  },
         {      36,     Trait::TYPE_SET,        "cImage1",            "miterLimit",         clsInvoke,      36,     0                  },
         {      37,     Trait::TYPE_FUN,        "cImage1",            "arc",                clsInvoke,      37,     0                  },
         {      38,     Trait::TYPE_FUN,        "cImage1",            "clip",               clsInvoke,      38,     0                  },
         {      0,      Trait::TYPE_END,        "cImage1",            0,                    0,              0,      0                  },
      };

      return TRAITS;
}

Trait* cImage1::extTraits()
{
      static Trait TRAITS[] = {
         {      0,      Trait::TYPE_BEGIN,      "cImage1",            0,                    0,              0,      0                  },
         {      0,      Trait::TYPE_FUN,        "cO3",                "image",              extInvoke,      0,      0                  },
         {      0,      Trait::TYPE_FUN,        "cO3",                "image",              extInvoke,      1,      0                  },
         {      0,      Trait::TYPE_END,        "cImage1",            0,                    0,              0,      0                  },
      };

      return TRAITS;
}

siEx cImage1::clsInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cImage1* pthis1 = (cImage1*) pthis;

      switch(index) {
         case 0:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( mode )");
            *rval = pthis1->mode();
            break;
         case 1:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( x )");
            *rval = pthis1->x();
            break;
         case 2:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( y )");
            *rval = pthis1->y();
            break;
         case 3:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( width )");
            *rval = pthis1->width();
            break;
         case 4:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( height )");
            *rval = pthis1->height();
            break;
         case 5:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( clear )");
            pthis1->clear(argv[0].toInt32());
            break;
         case 6:
            if (argc != 3)
               return o3_new(cEx)("Invalid argument count. ( setPixel )");
            pthis1->setPixel(argv[0].toInt32(),argv[1].toInt32(),argv[2].toInt32());
            break;
         case 7:
            if (argc != 2)
               return o3_new(cEx)("Invalid argument count. ( getPixel )");
            *rval = pthis1->getPixel(argv[0].toInt32(),argv[1].toInt32());
            break;
         case 8:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( src )");
            *rval = siFs(pthis1->src(siFs (argv[0].toScr()),&ex));
            break;
         case 9:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( savePng )");
            *rval = pthis1->savePng(siFs (argv[0].toScr()),&ex);
            break;
         case 10:
            if (argc != 5)
               return o3_new(cEx)("Invalid argument count. ( rect )");
            pthis1->rect(argv[0].toInt32(),argv[1].toInt32(),argv[2].toInt32(),argv[3].toInt32(),argv[4].toInt32());
            break;
         case 11:
            if (argc != 5)
               return o3_new(cEx)("Invalid argument count. ( line )");
            pthis1->line(argv[0].toInt32(),argv[1].toInt32(),argv[2].toInt32(),argv[3].toInt32(),argv[4].toInt32());
            break;
         case 12:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( decodeColor )");
            *rval = pthis1->decodeColor(argv[0].toStr());
            break;
         case 13:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( fillStyle )");
            pthis1->fillStyle(argv[0].toStr());
            break;
         case 14:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( strokeStyle )");
            pthis1->strokeStyle(argv[0].toStr());
            break;
         case 15:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( strokeWidth )");
            pthis1->strokeWidth(argv[0].toDouble());
            break;
         case 16:
            if (argc != 4)
               return o3_new(cEx)("Invalid argument count. ( clearRect )");
            pthis1->clearRect(argv[0].toDouble(),argv[1].toDouble(),argv[2].toDouble(),argv[3].toDouble());
            break;
         case 17:
            if (argc != 4)
               return o3_new(cEx)("Invalid argument count. ( fillRect )");
            pthis1->fillRect(argv[0].toDouble(),argv[1].toDouble(),argv[2].toDouble(),argv[3].toDouble());
            break;
         case 18:
            if (argc != 4)
               return o3_new(cEx)("Invalid argument count. ( strokeRect )");
            pthis1->strokeRect(argv[0].toDouble(),argv[1].toDouble(),argv[2].toDouble(),argv[3].toDouble());
            break;
         case 19:
            if (argc != 2)
               return o3_new(cEx)("Invalid argument count. ( moveTo )");
            pthis1->moveTo(argv[0].toDouble(),argv[1].toDouble());
            break;
         case 20:
            if (argc != 2)
               return o3_new(cEx)("Invalid argument count. ( lineTo )");
            pthis1->lineTo(argv[0].toDouble(),argv[1].toDouble());
            break;
         case 21:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( closePath )");
            pthis1->closePath();
            break;
         case 22:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( beginPath )");
            pthis1->beginPath();
            break;
         case 23:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( fill )");
            pthis1->fill();
            break;
         case 24:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( stroke )");
            pthis1->stroke();
            break;
         case 25:
            if (argc != 4)
               return o3_new(cEx)("Invalid argument count. ( quadraticCurveTo )");
            pthis1->quadraticCurveTo(argv[0].toDouble(),argv[1].toDouble(),argv[2].toDouble(),argv[3].toDouble());
            break;
         case 26:
            if (argc != 6)
               return o3_new(cEx)("Invalid argument count. ( bezierCurveTo )");
            pthis1->bezierCurveTo(argv[0].toDouble(),argv[1].toDouble(),argv[2].toDouble(),argv[3].toDouble(),argv[4].toDouble(),argv[5].toDouble());
            break;
         case 27:
            if (argc != 2)
               return o3_new(cEx)("Invalid argument count. ( translate )");
            pthis1->translate(argv[0].toDouble(),argv[1].toDouble());
            break;
         case 28:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( rotate )");
            pthis1->rotate(argv[0].toDouble());
            break;
         case 29:
            if (argc != 2)
               return o3_new(cEx)("Invalid argument count. ( scale )");
            pthis1->scale(argv[0].toDouble(),argv[1].toDouble());
            break;
         case 30:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( save )");
            pthis1->save();
            break;
         case 31:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( restore )");
            pthis1->restore();
            break;
         case 32:
            if (argc != 6)
               return o3_new(cEx)("Invalid argument count. ( setTransform )");
            pthis1->setTransform(argv[0].toDouble(),argv[1].toDouble(),argv[2].toDouble(),argv[3].toDouble(),argv[4].toDouble(),argv[5].toDouble());
            break;
         case 33:
            if (argc != 6)
               return o3_new(cEx)("Invalid argument count. ( transform )");
            pthis1->transform(argv[0].toDouble(),argv[1].toDouble(),argv[2].toDouble(),argv[3].toDouble(),argv[4].toDouble(),argv[5].toDouble());
            break;
         case 34:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( lineCap )");
            pthis1->lineCap(argv[0].toStr());
            break;
         case 35:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( lineJoin )");
            pthis1->lineJoin(argv[0].toStr());
            break;
         case 36:
            if (argc != 1)
               return o3_new(cEx)("Invalid argument count. ( miterLimit )");
            pthis1->miterLimit(argv[0].toDouble());
            break;
         case 37:
            if (argc != 6)
               return o3_new(cEx)("Invalid argument count. ( arc )");
            pthis1->arc(argv[0].toDouble(),argv[1].toDouble(),argv[2].toDouble(),argv[3].toDouble(),argv[4].toDouble(),argv[5].toBool());
            break;
         case 38:
            if (argc != 0)
               return o3_new(cEx)("Invalid argument count. ( clip )");
            pthis1->clip();
            break;
      }
      return ex;
}

siEx cImage1::extInvoke(iScr* pthis, iCtx* ctx, int index, int argc,
           const Var* argv, Var* rval)
{
      siEx ex;
      cImage1* pthis1 = (cImage1*) pthis;

      switch(index) {
         case 0:
            if (argc==0) {
               *rval = pthis1->image();
            }
            else if(2 <= argc && 3 >= argc) {
               *rval = pthis1->image(argv[0].toInt32(),argv[1].toInt32(),argc > 2 ? argv[2].toStr() : "argb");
            }
            else{
               return o3_new(cEx)("Invalid argument count.");
            }
            break;
      }
      return ex;
}

}
#endif
#pragma warning(default : 4100)
#pragma warning(default : 4189)
