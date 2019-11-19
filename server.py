from db_connect import DB_CONN

import os
import numpy as np
import pandas as pd
import threading
import requests
import traceback

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from items import *

import pickle as pk

def load_from_pk(path):
    with open(path,'rb') as f:
        ret= pk.load(f)
    print('load from',path)
    return ret

def save_to_pk(path,obj):
    with open(path,'wb') as f:
        pk.dump(obj,f)
    print('save to',path)

base_dir = os.path.abspath(os.path.dirname(__file__))

def getTreeRule(tree,path,data,rule):
    for nd in np.where(path)[1]:
        feature=tree.feature[nd]
        if( feature==-2):
            break
        (down,up)=rule.get(tree.feature[nd],(-100000,100000))
        d=data[tree.feature[nd]]
        th=tree.threshold[nd]

        if(d>=th and th>down):
            rule[tree.feature[nd]]=(th,up)
        elif(d<th and th<up):
            rule[tree.feature[nd]]=(down,th)
    return rule

def get_trees_rules(rf,data):
    rules={}
    for estimator in rf.estimators_:
        tree=estimator.tree_
        path = tree.decision_path(data).toarray()
        rules=getTreeRule(tree,path,data[0],rules)
    return rules

class Loader():
    def __init__(self,DB_CONN):
        data,self.label=DB_CONN.getAllLabeledRecord(tbname_yq='chongqing_yunqianjianchabiao_2017',
        tbname_jj='chongqing_renshenjiejubiao_2017')
        df = pd.DataFrame.from_dict(data)
        df.fillna(value=0, inplace=True)
        df = df.replace(r'\s*', 0, regex=True)
        X=[]
        for ind,d in df.iterrows():
            tmp=[]
            for itm in ITEMS_LEARN:
                tmp.append(d[itm])
            X.append(tmp)

        self.X=X
        print('loaded',len(self.label))

    def get_18(self,DB_CONN):
        data,label=DB_CONN.getAllLabeledRecord()
        df = pd.DataFrame.from_dict(data)
        df.fillna(value=0, inplace=True)
        df = df.replace(r'\s*', 0, regex=True)
        X=[]
        for ind,d in df.iterrows():
            tmp=[]
            for itm in ITEMS_LEARN:
                tmp.append(d[itm])
            X.append(tmp)

        print('loaded',len(label))
        return X,label


class Prob_from_rules():
    def __init__(self,Loader):
        self.X=loader.X
        self.label=loader.label

    def get_prob(self,rules_dict):
        def test_data(dt,rules):
            for feature,r in rules.items():
                if(dt[feature]<r[1]and dt[feature]>=r[0]):
                    pass
                else:
                    return False
            return True

        label=self.label
        ill_cnt=0
        ill_when_sat=0
        sat=0
        all_cnt=0

        for ind,dt in enumerate(self.X):
            lb=label[ind]
            flg=test_data(dt,rules_dict)
            if(lb==1):
                ill_cnt=ill_cnt+1

            if(flg):
                sat=sat+1

            if(flg and lb==1):
                ill_when_sat=ill_when_sat+1
            all_cnt=all_cnt+1
        prob=ill_when_sat/sat
        return prob,ill_when_sat,sat,all_cnt,ill_cnt

    def get_prob_bys(self,rules_dict):
        X=self.X
        labels=self.label
        #总样本数
        all_cnt=len(labels)
        #不良样本数
        ill_cnt=0
        for lb in labels:
            if(lb == 1):
                ill_cnt+=1
        cnt_dict={}
        ill_cnt_dict={}
        for ind,dt in enumerate(X):
            lb=labels[ind]
            for f,r in rules_dict.items():
                if(dt[f]>=r[0] and dt[f]<r[1]):
                    cnt=cnt_dict.get(f,0)
                    cnt_dict[f]=cnt+1
                    if(lb==1):
                        cnt=ill_cnt_dict.get(f,0)
                        ill_cnt_dict[f]=cnt+1

        p_cnd=1
        for f,c in ill_cnt_dict.items():
            p_cnd*=(c+1)/(ill_cnt+2)

        p_ill=ill_cnt/all_cnt 

        p_all=1
        for f,c in cnt_dict.items():
            p_all*=(c+1)/(all_cnt+2)

        return p_ill*p_cnd/p_all,p_ill,p_cnd,p_all,ill_cnt_dict,cnt_dict


class Model:
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=1000,criterion='gini',max_features='sqrt',
            n_jobs=-1,oob_score=True,class_weight='balanced')
        self.column_order=ITEMS_LEARN


    def training(self,loader):
        X,label=loader.X,loader.label
        self.rf.fit(X,label)
        # cvs=cross_val_score(self.rf,X,label,cv=10)
        # print('cvs =',cvs)
        print('score=',self.rf.score(X,label))
        print('oob_score=',self.rf.oob_score_)

    def get_importance(self, df):
        importance_all = np.zeros(self.rf.n_features_)
        for estimator in self.rf.estimators_:
            importance = np.zeros(self.rf.n_features_)
            tree = estimator.tree_
            path = tree.decision_path(np.float32(df.values)).toarray()
            for node in np.where(path)[1]:
                left = tree.children_left[node]
                right = tree.children_right[node]
                importance[tree.feature[node]] += (
                    tree.impurity[node] -
                    tree.impurity[left] -
                    tree.impurity[right])
            importance_all += importance / importance.sum()
        return importance_all

    def get_rule(self,df):
        rules=get_trees_rules(self.rf,np.float32(df.values))
        return rules


    def predict(self,df):
        df = pd.DataFrame.from_dict([df])
        df.fillna(value=0, inplace=True)
        data1 = df
        data1 = data1[self.column_order]
        # 临时救场
        data1 = data1.replace(r'\s*', 0, regex=True)
        result = self.rf.predict_proba(data1)
        rules=self.get_rule(data1)
        importance=self.get_importance(data1)
        importance = sorted(enumerate(importance), key=lambda x:-x[1])
        return result[0],rules,importance,self.rf.predict(data1)

class RandomForestServer(threading.Thread):

    def __init__(self, queue):
        super(RandomForestServer, self).__init__()
        self.daemon = True
        self.queue = queue
        self.model = Model()

    def run(self):
        while True:
            try:
                record_id = self.queue.get()
                self.model.predict(record_id)
            except Exception:
                print(traceback.format_exc())


def test_data(dt,rules):
    for feature,r in rules.items():
        feature=ITEMS_LEARN[feature]
        if(dt[feature]<r[1]and dt[feature]>=r[0]):
            pass
        else:
            print(feature,dt[feature],r)
            return False
    return True

if __name__=="__main__":
    loader=Loader(DB_CONN)
    save_to_pk('loader.pk',loader)
    loader=load_from_pk('loader.pk')
    
    model=Model()
    model.training(loader)
    save_to_pk('rf.pk',model)
    model=load_from_pk('rf.pk')
    
    prober=Prob_from_rules(loader)
    data_18,label_18=loader.get_18(DB_CONN)
    print('test score=',model.rf.score(data_18,label_18))

    tables=[
    'beijing_yunqianjianchabiao_2017',
    'beijing_yunqianjianchabiao_2018',
    'chongqing_yunqianjianchabiao_2017',
    'chongqing_yunqianjianchabiao_2018',
    'fujian_yunqianjianchabiao_2017',
    'guangdong_yunqianjianchabiao_2017',
    'guangdong_yunqianjianchabiao_2018',
    'henan_yunqianjianchabiao_2017',
    'henan_yunqianjianchabiao_2018',
    'liaoning_yunqianjianchabiao_2017',
    'liaoning_yunqianjianchabiao_2018',
    'ningxia_yunqianjianchabiao_2017',
    'shandong_yunqianjianchabiao_2017',
    'shandong_yunqianjianchabiao_2018',
    'shanghai_yunqianjianchabiao_2017',
    'tianjin_yunqianjianchabiao_2017',
    'tianjin_yunqianjianchabiao_2018',
    'zhejiang_yunqianjianchabiao_2017',
    ]
    lis=[]
    table=tables[0]
    ok=0
    err=0
    for record,label in DB_CONN.getEnumerateRecord():
        r,rules,importance,predicted=model.predict(record)
        # lis.append(r)
        # if r>0.5 and label==0 or r<0.5 and label==1:
        #     print(rules)
        #     print(r,label)
        #     err=err+1
        # else:
        #     print(f'ok {ok}')
        #     ok=ok+1
        rule_list={}
        ind=0
        print(r,label,predicted)
        for f,val in importance:
            rule_list[f]=rules.get(f,(-10000,10000))
            print(prober.get_prob_bys(rule_list))
            ind=ind+1
            if(ind>=10):
                break
        holder=input('stoped')
        print('\n\n')

    # print(err,ok)


