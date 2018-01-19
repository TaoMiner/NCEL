from ncel.data.kbp15_formatter import *
from ncel.data.kbp10_formatter import *

LANG = ('eng', 'cmn', 'spa')
DOC_TYPE = ('nw','df')
def getKBPRootPath():
    return '/home/caoyx/data/kbp/LDC2017E03_TAC_KBP_Entity_Discovery_and_Linking_Comprehensive_Training_and_Evaluation_Data_2014-2016/data/'

def getKBPOutRootPath():
    return '/home/caoyx/data/kbp/kbp_cl/'

def getKbpOutTextPath(year, isEval, lang, docType):
    tmp_path = getKBPOutRootPath()
    tmp_path += 'kbp15/' if year == 2015 else 'kbp16/'
    tmp_path += lang + '/'
    tmp_path += 'eval/' if isEval else 'train/'
    tmp_path += docType + '/'
    return tmp_path

def getKbpOutXmlFile(year, isEval, lang, docType):
    tmp_path = getKBPOutRootPath()
    tmp_path += 'kbp15/' if year == 2015 else 'kbp16/'
    tmp_path += lang + '/'
    tmp_path += 'eval/' if isEval else 'train/'
    tmp_filename = 'kbp15_' if year == 2015 else 'kbp16_'
    tmp_filename += docType + '_gold.xml'
    return tmp_path + tmp_filename

def getKBPQueryFile(year, isEval):
    tmp_path = '/eval' if isEval else '/training'
    tmp_type = 'evaluation' if isEval else 'training'
    tmp_edl = '_tedl_' if year == 2015 else '_edl_'
    return getKBPRootPath() + str(year) + tmp_path + '/tac_kbp_' + str(year) + tmp_edl + tmp_type + '_gold_standard_entity_mentions.tab'

def getKBPDataPath(year, isEval, lang, docType):
    if year == 2015:
        tmp_path = '/eval' if isEval else '/training'
        docTypePath = '/newswire/' if docType == DOC_TYPE[0] else '/discussion_forum/'
    else:
        tmp_path = '/eval/'
        docTypePath = '/nw/' if docType == DOC_TYPE[0] else '/df/'
    tmp_source = '/source_documents/' if isEval else '/source_docs/'

    return getKBPRootPath() + str(year) + tmp_path + tmp_source + lang + docTypePath


def formatKBP15(year, isEval, lang, docType):
    query_file = getKBPQueryFile(year, isEval)
    text_path = getKBPDataPath(year, isEval, lang, docType)
    kf = kbp15Formatter(text_path, query_file, str(year)+docType, lang=lang)
    kf.format(getKbpOutTextPath(year, isEval, lang, docType),
              getKbpOutXmlFile(year, isEval, lang, docType))

def formatKBP10():
    text_path = '/home/caoyx/data/kbp/kbp2010/TAC_2010_KBP_Source_Data/data/2010/wb/'
    query_xml = '/home/caoyx/data/kbp/kbp2010/TAC_2010_KBP_Evaluation_Entity_Linking_Gold_Standard_V1.0/data/tac_2010_kbp_evaluation_entity_linking_queries.xml'
    query_ans_file = '/home/caoyx/data/kbp/kbp2010/TAC_2010_KBP_Evaluation_Entity_Linking_Gold_Standard_V1.0/data/tac_2010_kbp_evaluation_entity_linking_query_types.tab'
    kf = kbp10Formatter(text_path, query_xml, query_ans_file)
    kf.format('/home/caoyx/data/kbp/kbp_cl/kbp10/eval/', '/home/caoyx/data/kbp/kbp_cl/kbp10/kbp10_eval_gold.xml')

if __name__ == "__main__":
    for year in [2015, 2016]:
        for isEval in [True, False]:
            if year == 2016 and not isEval : continue
            for lang in LANG:
                for docType in DOC_TYPE:
                    print("preprocessing {0}, {1}, {2}, {3}".format(year, 'eval' if isEval else 'train', lang, docType))
                    formatKBP15(year, isEval, lang, docType)