#In the next recipe, we'll look at how to tune the random forest classifier.
#Let's start by importing datasets:

from sklearn import datasets
import numpy as np
X, y = datasets.make_classification(1000)

# X(1000,20)
#y(1000) 取值范围【0,1】

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200,criterion='gini',max_features='log2',max_depth=20,
    min_samples_split=0.02,min_samples_leaf=0.01,n_jobs=-1)

print(X)
print(y)

rf.fit(X, y)
print ("Accuracy:\t", (y == rf.predict(X)).mean())
print ("Total Correct:\t", (y == rf.predict(X)).sum())

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
        print(nd,tree.feature[nd],down,up,d,th,rule[tree.feature[nd]],tree.children_left[nd],tree.children_right[nd])
    return rule

def get_trees(rt,data):
    data=np.float32([data])
    rules={}
    for estimator in rf.estimators_:
        tree=estimator.tree_
        path = tree.decision_path(data).toarray()
        rules=getTreeRule(tree,path,data[0],rules)
    print(rules)

print(y)
data=400
get_trees(rf,X[data])
print(rf.predict_proba([X[data]]))
print(rf.predict([X[data]]))
print(y[data])
exit()


#每个例子属于哪个类的概率
probs = rf.predict_proba(X)
import pandas as pd
probs_df = pd.DataFrame(probs, columns=['0', '1'])
probs_df['was_correct'] = rf.predict(X) == y
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(7, 5))
probs_df.groupby('0').was_correct.mean().plot(kind='bar', ax=ax)
ax.set_title("Accuracy at 0 class probability")
ax.set_ylabel("% Correct")
ax.set_xlabel("% trees for 0")
plt.show()

#检测重要特征
# rf = RandomForestClassifier()
# rf.fit(X, y)
f, ax = plt.subplots(figsize=(7, 5))
ax.bar(range(len(rf.feature_importances_)),rf.feature_importances_)
ax.set_title("Feature Importances")
plt.show()
