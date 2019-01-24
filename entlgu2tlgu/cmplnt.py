# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 12:27:26 2018

@author: user
"""

import requests 
import codecs
import re
untranslated = codecs.open('problems.txt','r','utf8')
content=untranslated.read()
print('\n')
print("Input :",content)
for i in content:
    sentences = re.split(r' *[\.\?!]',content)
vr=len(sentences)    
for i in range(vr-1):
    base_url = 'https://www.google.com/inputtools/request?text='+sentences[i]+'&ime=transliteration_en_te'
    json_data = requests.get(base_url).json()
    resu = json_data[1][0][1]
    result='\n'.join(resu)
    print(result)
    file=codecs.open('newl.txt', 'a+' ,'utf-8').write(result+".")


import codecs
from google.cloud import translate
client = translate.Client()
untranslated = codecs.open('newl.txt','r','utf8')
content=untranslated.read()
import re
print('\n')
nstr = re.sub(r'[?|$|!|"|,|#]',r'',content)
print(nstr)
print('\n')
print("Actual :",content)
#trans=client.translate(content,target_language='en')
#
#translated_text=u'{}'.format(trans['translatedText'])
#print( translated_text+"\n")
#
#file=codecs.open('senteng.txt', 'w').write(translated_text+".")
for stuff in sentences:
     print(stuff)
     trans=client.translate(stuff,target_language='en')
     translated_text=u'{}'.format(trans['translatedText'])
     print(translated_text + "\n")
     file=codecs.open('transl.txt', 'a+').write(translated_text+"\n")