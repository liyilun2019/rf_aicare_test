import MySQLdb
import sys;
sys.path.append("../")
sys.path.append('../..')
from items import *
import pandas as pd
from datetime import datetime,date
from passwords import *

items=ITEMS_LEARN

# for ind,item in enumerate(items):
#     print(f"{ind} == {ITEMS.index(item)}:\t{item}")
# exit()

jjtable_items=['pre_result_type_zhengchanghuochan',
'pre_result_type_zaochan',
'pre_result_type_dichushengtizhong',
'pre_result_type_chushengquexian',
'pre_result_type_ziranliuchang',
'pre_result_type_yixuexingrengongliuchang',
'pre_result_type_zhiliaoxingyinchan',
'pre_result_type_yiweirenshen',
'pre_result_type_sitaisichan',
'pre_result_type_guoqirenshen',
'pre_result_type_judaer',
'pre_result_type_qita',]

table_vars=''
for ind,i in enumerate(items):
    table_vars+=f'{i}'
    if not (ind == len(items)-1):
        table_vars+=','

class Record_DB():
    def __init__(self):
        self.db = MySQLdb.connect(host,name,key,db, charset='utf8' )
        self.cursor=self.db.cursor()

    def getRecord(self,recordid,tbname='beijing_yunqianjianchabiao_2018'):
        print(f"select {table_vars} from {tbname} where archive_code='{recordid}';")
        self.cursor.execute(f"select {table_vars} from {tbname} where archive_code='{recordid}';")
        rows=self.cursor.fetchall()
        ret={}
        # print(rows==()) # ()代表空
        for r in rows:
            for ind,itm in enumerate(items):
                ret[itm]=r[ind]
            return ret

    def getEnumerateRecord(self,tbname_yq='beijing_yunqianjianchabiao_2018',
        tbname_jj='beijing_renshenjiejubiao_2018'):
        table_vars=''
        for ind,i in enumerate(items):
            table_vars+=f'IFNULL(a.{i},0)'
            if not (ind == len(items)-1):
                table_vars+=','

        jjtable_vars=''
        for ind,i in enumerate(jjtable_items):
            jjtable_vars+=f'IFNULL(b.{i},0)'
            if not (ind == len(jjtable_items)-1):
                jjtable_vars+=','
        jjtable_sum=''
        for ind,i in enumerate(jjtable_items):
            jjtable_sum+=f'IFNULL(b.{i},0)'
            if not (ind == len(jjtable_items)-1):
                jjtable_sum+='+'

        self.cursor.execute(f"select {table_vars},{jjtable_vars},{jjtable_sum} from {tbname_yq} a join {tbname_jj} b on a.archive_code=b.archive_code limit 10000;")
        r=self.cursor.fetchone()
        while r is not None:
            tmp={}
            lb={}
            for ind,itm in enumerate(items):
                tmp[itm]=r[ind]
            for ind,itm in enumerate(jjtable_items):
                lb[itm]=r[ind+len(items)]
            lb['sum']=r[len(items)+len(jjtable_items)]
            
            if(lb['sum']==0):
                pass
            elif(lb['pre_result_type_zhengchanghuochan']==0):
                yield tmp,1
            else:
                yield tmp,0
            r=self.cursor.fetchone()


    def getAllLabeledRecord(self,tbname_yq='beijing_yunqianjianchabiao_2017',
        tbname_jj='beijing_renshenjiejubiao_2017'):
        table_vars=''
        for ind,i in enumerate(items):
            table_vars+=f'IFNULL(a.{i},0)'
            if not (ind == len(items)-1):
                table_vars+=','

        jjtable_vars=''
        for ind,i in enumerate(jjtable_items):
            jjtable_vars+=f'IFNULL(b.{i},0)'
            if not (ind == len(jjtable_items)-1):
                jjtable_vars+=','
        jjtable_sum=''
        for ind,i in enumerate(jjtable_items):
            jjtable_sum+=f'IFNULL(b.{i},0)'
            if not (ind == len(jjtable_items)-1):
                jjtable_sum+='+'

        self.cursor.execute(f"select {table_vars},{jjtable_vars},{jjtable_sum} from {tbname_yq} a join {tbname_jj} b on a.archive_code=b.archive_code;")
        r=self.cursor.fetchone()
        X=[]
        y=[]
        while r is not None:
            tmp={}
            lb={}
            for ind,itm in enumerate(items):
                tmp[itm]=r[ind]
            for ind,itm in enumerate(jjtable_items):
                lb[itm]=r[ind+len(items)]
            lb['sum']=r[len(items)+len(jjtable_items)]
            
            if(lb['sum']==0):
                pass
            elif(lb['pre_result_type_zhengchanghuochan']==0): #异常
                y.append(1)
                X.append(tmp)
            else:
                y.append(0)
                X.append(tmp)
            r=self.cursor.fetchone()
        return X,y

    def getAllRecordID(self,tbname='beijing_yunqianjianchabiao_2018'):
        ret=[]
        self.cursor.execute(f"select archive_code from {tbname} limit 10;")
        rows=self.cursor.fetchall()
        for r in rows:
            ret.append(r[0])
        return ret

    def getAllRecordSummary(self,tbname='beijing_yunqianjianchabiao_2018'):
        ret=[]
        self.cursor.execute(f"select archive_code,input_date_archive_ymd,doctor_id_w,service_organization_name from {tbname} limit 10000;")
        rows=self.cursor.fetchall()
        for row in rows:
            ret.append({
                'service_code':row[0],
                'create_date':row[1],
                'doctor_id':row[2],
                'service_organization_name':row[3],
                })
        return ret

    def showTables(self):
        self.cursor.execute(f'show tables')
        rows=self.cursor.fetchall()
        for row in rows:
            if("yunqianjianchabiao_2018" in row[0]) or ('yunqianjianchabiao_2017' in row[0]):
                print(f"'{row[0]}',")


DB_CONN=Record_DB()

if __name__=='__main__':
    db=DB_CONN
    data,label=db.getAllLabeledRecord()
    df={}
    for d in data:
        for (k,v) in d.items():
            lis=df.get(k,[])
            lis.append(v)
            df[k]=lis
    df=pd.DataFrame(df)
    df.to_csv('out2.csv',encoding='gbk')

    # for i in items:
    #     if(''in i):
    #         print(f"'{i}',")

