# -*- coding: utf-8 -*-
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

class xmlHandler(object):
    def __init__(self, txt_elem_list, int_elem_list):
        self.doc_name = ""
        self.annotations = []
        self._txt_elem_list = txt_elem_list
        self._int_elem_list = int_elem_list

    def process(self, filename):
        path = []
        annotation = {}
        for event, elem in ET.iterparse(filename, events=("start", "end")):
            if event == 'start':
                if elem.tag == 'document':
                    self.doc_name = elem.attrib['docName']
                    path.append(elem.tag)
                elif elem.tag == 'annotation':
                    annotation = {}
                    path.append(elem.tag)
            if event == 'end':
                # process the tag
                if 'document' in path:
                    if elem.tag == 'document' and len(self.doc_name) > 0 and len(self.annotations) > 0:
                        yield self.doc_name, self.annotations
                        del self.annotations[:]
                        path.pop()
                        elem.clear()  # disgrad elem
                    if 'annotation' in path:
                        if elem.tag in self._txt_elem_list:
                            annotation[elem.tag] = elem.text
                        elif elem.tag in self._int_elem_list:
                            annotation[elem.tag] = int(elem.text)
                        elif elem.tag == 'annotation':
                            self.annotations.append(annotation)
                            path.pop()

class kbp10XmlHandler():
    def __init__(self, xml_file):
        self._xml_file = xml_file
        self.doc_id = ''
        self.name = ''

    def queries(self):
        for event, elem in ET.iterparse(self._xml_file):
            if event == 'end':
                # process the tag
                if elem.tag == 'name':
                    self.name = elem.text
                elif elem.tag == 'docid':
                    self.doc_id = elem.text
                elif elem.tag == 'query':
                    yield elem.attrib['id'], self.doc_id, self.name

class kbp15XmlHandler():
    def __init__(self, xml_file):
        self._xml_file = xml_file
        self._tmp_file = xml_file + '.tmp'

    def texts(self):
        doc = ' '.join(self.originalText())
        fout = open(self._tmp_file, 'w', encoding='utf-8')
        fout.write(doc)
        fout.close()
        for event, elem in ET.iterparse(self._tmp_file):
            if event == 'end':
                yield elem.text

    def originalText(self):
        for event, elem in ET.iterparse(self._xml_file):
            if event == 'end':
                # process the tag
                if elem.tag == 'ORIGINAL_TEXT':
                    yield elem.text

def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def buildXml(xml_file, doc_mentions):
    root = ET.Element("KBP2010.entityAnnotation")
    for doc_id in doc_mentions:
        doc_elem = ET.SubElement(root, "document")
        doc_elem.set("docName", doc_id)
        for m in doc_mentions[doc_id]:
            annotation_elem = ET.SubElement(doc_elem, "annotation")
            mention_elem = ET.SubElement(annotation_elem, "mention")
            mention_elem.text = m[0]
            wikiName_elem = ET.SubElement(annotation_elem, "wikiId")
            wikiName_elem.text = m[1]
            offset_elem = ET.SubElement(annotation_elem, "offset")
            offset_elem.text = str(m[2])
            length_elem = ET.SubElement(annotation_elem, "length")
            length_elem.text = str(len(m[0]))
    indent(root)
    tree = ET.ElementTree(root)
    tree.write(xml_file, encoding="UTF-8")

if __name__ == "__main__":
    # Demo:
    xh = xmlHandler(['mention', 'wikiName'], ['offset', 'length'])
    for doc_name, mentions in xh.process("/Users/ethan/Downloads/WNED/wned-datasets/ace2004/ace2004.xml"):
        print(doc_name)
        print(mentions)