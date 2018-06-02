import requests
import openpyxl

import os
dirPrefix = os.path.join(os.sep, 'mnt')

url = 'https://assets.digitalocean.com/articles/eng_python/beautiful-soup/mockturtle.html'

print ("Hello World2!")

page = requests.get(url)

print ("Status code: %s" % page.status_code)

print (page.text)

wb = openpyxl.Workbook()


wb.save(filename=os.path.join(dirPrefix, 'test.xlsx'))
with open(os.path.join(dirPrefix, 'testfile.txt'), 'w') as f:
    pass