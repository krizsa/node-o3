#ifndef O3_APPLE
#define LIBXML_SAX1_ENABLED 1
#define LIBXML_XPATH_ENABLED 1
#define LIBXML_OUTPUT_ENABLED 1
#endif // O3_APPLE

#include <libxml/parser.h>
#include <libxml/xpath.h>
#include <libxml/xpathInternals.h>
#include <libxml/tree.h>

#include "o3_iXml1.h"
#include "o3_cXmlNodeList1.h"
#include "o3_cXmlNamedNodeMap1.h"
#include "o3_cXmlNode1.h"
#include "o3_cXmlAttr1.h"
#include "o3_cXmlNodeArray1.h"
#include "o3_cXmlElement1.h"
#include "o3_cXmlCharacterData1.h"
#include "o3_cXmlText1.h"
#include "o3_cXmlComment1.h"
#include "o3_cXmlCDATASection1.h"
#include "o3_cXmlDocument1.h"
#include "o3_cXml1.h"

#ifdef O3_WITH_GLUE
#include "o3_scr_cXmlNodeList1.h"
#include "o3_scr_cXmlNamedNodeMap1.h"
#include "o3_scr_cXmlNode1.h"
#include "o3_scr_cXmlAttr1.h"
#include "o3_scr_cXmlNodeArray1.h"
#include "o3_scr_cXmlElement1.h"
#include "o3_scr_cXmlCharacterData1.h"
#include "o3_scr_cXmlText1.h"
#include "o3_scr_cXmlComment1.h"
#include "o3_scr_cXmlCDATASection1.h"
#include "o3_scr_cXmlDocument1.h"
#include "o3_scr_cXml1.h"
#endif