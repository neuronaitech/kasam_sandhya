# -*- coding: utf-8 -*-

#######################TESSERACT
import codecs
from PIL import Image
from pytesseract import image_to_string
result= print(image_to_string(Image.open('D:\\datafiles\\dddd.jpg'), lang='eng'))
resul=image_to_string(Image.open('D:\\datafiles\\tlguu.png'),lang='tel')
file=codecs.open('img2.txt', 'w' ,'utf_8').write(resul)
from google.cloud import translate
client = translate.Client()
untranslated = codecs.open('img2.txt','r','utf8')
content=untranslated.read()
print('\n')
print("Actual :",content)
import re
nstr = re.sub(r'[?|$|!|"|#]',r'',content)
print(nstr)
print('\n')
trans=client.translate(content,target_language='en')
translated_text=u'{}'.format(trans['translatedText'])
print("After  : ", translated_text)

file=codecs.open('senteng.txt', 'w').write(translated_text)
