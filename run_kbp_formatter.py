from ncel.data.kbp15_formatter import *
from ncel.data.kbp10_formatter import *

def formatKBP15():
    query_file = '/home/caoyx/data/kbp/LDC2017E03_TAC_KBP_Entity_Discovery_and_Linking_Comprehensive_Training_and_Evaluation_Data_2014-2016/data/2015/eval/tac_kbp_2015_tedl_evaluation_gold_standard_entity_mentions.tab'
    text_path = '/home/caoyx/data/kbp/LDC2017E03_TAC_KBP_Entity_Discovery_and_Linking_Comprehensive_Training_and_Evaluation_Data_2014-2016/data/2015/eval/source_documents/eng/discussion_forum/'
    kf = kbp15Formatter(text_path, query_file, DATA_TYPE[0], lang='eng')
    kf.format('/home/caoyx/data/kbp/kbp_cl/kbp15/eval/eng/df/',
              '/home/caoyx/data/kbp/kbp_cl/kbp15/eval/eng/kbp15_df.xml')

def formatKBP10():
    text_path = '/home/caoyx/data/kbp/kbp2010/TAC_2010_KBP_Source_Data/data/2010/wb/'
    query_xml = '/home/caoyx/data/kbp/kbp2010/TAC_2010_KBP_Evaluation_Entity_Linking_Gold_Standard_V1.0/data/tac_2010_kbp_evaluation_entity_linking_queries.xml'
    query_ans_file = '/home/caoyx/data/kbp/kbp2010/TAC_2010_KBP_Evaluation_Entity_Linking_Gold_Standard_V1.0/data/tac_2010_kbp_evaluation_entity_linking_query_types.tab'
    kf = kbp10Formatter(text_path, query_xml, query_ans_file)
    kf.format('/home/caoyx/data/kbp/kbp_cl/kbp10/eval/', '/home/caoyx/data/kbp/kbp_cl/kbp10/kbp10_eval_gold.xml')

if __name__ == "__main__":
    formatKBP10()