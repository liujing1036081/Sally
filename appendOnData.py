# encoding=utf-8
#为irony-sentences首尾添加了特定的符号|NULL|和|5|5|0|0和ID

import re
#pchinese=re.compile('([\u4e00-\u9fa5]+|[\u3000-\u303f\ufb00-\ufffd]|[\d]|[...])') #判断是否为中文的正则表达式
f=open("input-2468.txt",encoding='utf-8') #打开要提取的文件
fw=open("getdata2468.txt","w")#打开要写入的文件
i=0
for line in f.readlines():   # 循环读取要读取文件的每一行
            line = line.strip('\n')
            line = '|NULL|'+line+'|5|5|0|0'
            i = i+1
            line = str(i)+line
            fw.write(line)     # 写入文件
            fw.write("\n")     # 不同行的要换行

            print (line)
f.close()
fw.close()#打开的文件记得关闭哦!