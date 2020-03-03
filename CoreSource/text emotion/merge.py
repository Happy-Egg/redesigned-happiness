import logging
import os
import sys
import codecs
 
program = os.path.basename( sys.argv[0] )
logger = logging.getLogger( program )
logging.basicConfig( format='%(asctime)s: %(levelname)s: %(message)s' )
logging.root.setLevel( level=logging.INFO )
 
def getContent(fullname):
    f = codecs.open(fullname, 'r',  encoding="gbk", errors="ignore")
    lines = []
    for eachline in f:
        #eachline = eachline.decode('gbk','ignore').strip()
        eachline = eachline.strip()
        if eachline:#很多空行
            lines.append(eachline)
    f.close()
    #print(fullname, 'OK')
    return lines
 
inp = 'data/ChnSentiCorp_htl_ba_2000'
folders = ['neg', 'pos']
for foldername in folders:
    logger.info('running ' + foldername + ' files.')
    outp = '2000_' + foldername + '.txt'#输出文件
    output = codecs.open( os.path.join('data/ChnSentiCorp_htl_ba_2000', outp), 'w')
    i = 0
     
    rootdir = os.path.join(inp, foldername)
    for each_txt in os.listdir(rootdir):
        contents = getContent( os.path.join(rootdir, each_txt) )
        i = i + 1
        output.write(''.join(contents) + '\n' )
 
    output.close
    logger.info("Saved "+str(i)+" files.")