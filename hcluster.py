import csv
from nltk import PorterStemmer
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
import nltk
import sys



porter=PorterStemmer()
lemmatizer=WordNetLemmatizer()
vector = TfidfVectorizer()
N=0
newwordlist=[]
stoplist = dict()
dic=dict()

with open("C:/Users/PCS/Desktop/NEWS Headline Clustering - WOG/stopword.txt") as stop:
    stoplist = stop.read()
   # print("before split:  ",stoplist)
    stop.close()
    stoplist=nltk.word_tokenize(stoplist)
  #  print("after split:  ",stoplist)
itr=0
with open("C:/Users/PCS/Desktop/NEWS Headline Clustering - WOG/abcnews-date-text1.csv",mode="r",encoding="ANSI") as file1:

    readfile=csv.reader(file1)
    for row in readfile:
        newwordlist.append([])
        N += 1
        listword=row[1].split()
        if len(listword)==1:
            continue
        else:
           # print(N,"new row")
            for i in listword:
                if i not in stoplist:
                    put=True
                    if (put==True):
                        i=lemmatizer.lemmatize(i)
                        i=porter.stem(i)
 #                       print(i)
                        newwordlist[itr].append(i)


            itr+=1
itr=0
docnum=1

newlist=[]

newlist= [" ".join(x) for x in newwordlist]
#print("N:  ", N)
while(itr<N):
    itr1=len(newwordlist[itr])
    numitr = 0
    #print(itr1)
    while(numitr<itr1-1):
        if numitr+2<=itr1:
            key = newwordlist[itr][numitr]
            value=newwordlist[itr][numitr+1]
  #      print("key",key,"value is",value)
        if docnum not in dic:
            dic[docnum] = {}
            dic[docnum].setdefault(key, []).append(value)
        else:
            dic[docnum].setdefault(key, []).append(value)
        numitr+=1
    itr+=1
    docnum+=1

#print(dic)
word_sequence={}
for x in dic:
     list2=[]
     list=dic[x]
     for x2 in list:
         l=list[x2]
         for l3 in range(len(l)):
            u=x2 + ' ' + l[l3]
 #           print(x)
 #           print(u)
            list2.append(u)
     word_sequence[x]=list2
#for x in word_sequence:
#    print(word_sequence[x])


finallist=[*word_sequence.values()]
finallist= [" ".join(x) for x in finallist]
tfidf=vector.fit_transform(finallist)
distance = 1 - cosine_similarity(tfidf)


from sklearn.cluster import AgglomerativeClustering
max1=sys.maxsize
cluster2=0
for i in range(2,51):
     cluster = AgglomerativeClustering(n_clusters=i, affinity='euclidean', linkage='complete')
     cluster_labels=cluster.fit_predict(distance)
     silhouette_avg = silhouette_score(distance, cluster_labels)
     print("For n_clusters =", i,
           "The average silhouette_score is :", silhouette_avg)
     if silhouette_avg<max1:
         max1=silhouette_avg
         cluster2=i


#######MAKING DENDOGRAM
         
         
         
Z = linkage(distance, 'complete' )
fig, ax = plt.subplots(figsize=(15, 20))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('All DOCUMENTS ')
plt.ylabel('distance (COMPELETE)')
dendrogram(Z, orientation="right" ,labels=newlist, leaf_rotation=360)


plt.tight_layout()

plt.savefig('C:/Users/PCS/Desktop/NEWS Headline Clustering - WOG/compelete_cluster.png', dpi=200)

print("Maximum Cluster is" + str(cluster2))

from sklearn.metrics import silhouette_score



#######MAKING FOLDERS

import os
from scipy.cluster.hierarchy import cut_tree
nc=cluster2
f = "Folder"
folders = []
x=0
for i in range(1,nc+1):
    x +=1
    #print(x)
    fol = f + str(x)
    folders.append(fol)




root_path = r'C:/Users/PCS/Desktop/NEWS Headline Clustering - WOG/'
for folder in folders:
   if not os.path.exists(root_path+folder):
       os.mkdir(os.path.join(root_path,folder))


arr=(cut_tree(Z ,nc).T)


for i in folders:
    root = r'C:/Users/PCS/Desktop/NEWS Headline Clustering - WOG/'+i + '/' + str(1) + '.txt'
    #print(root)


doclen=(len(word_sequence))

k=1
for doc in newlist:
   #print("*************************",k , arr[0][k]) 
   if k != doclen: 
       
       x=arr[0][k] + 1
       #print(x)
       path = root_path+folders[x-1] + '/' + str(k+1) + '.txt'
       #print("filepath:   " , path)
       f= open(path,"w") 
       f.write(newlist[k])
       f.close()
       k+=1
    
fd= []
fd = [[1 for i in range(2)] for i in range(nc+1)]    

for i in range(1,14):
    fd[i][0]=i
    

from scipy.cluster.hierarchy import dendrogram, linkage  
from matplotlib import pyplot as plt



Z = linkage(fd, 'complete' )
fig, ax = plt.subplots(figsize=(15, 20))
plt.title('Hierarchical Clustering Dendrogram')
plt.ylabel('All DOCUMENTS ')
plt.xlabel('distance (COMPELETE)')
dendrogram(Z, orientation="right" ,labels=range(1, nc+2), leaf_rotation=360)


#plt.show() 

plt.tight_layout()

plt.savefig('C:/Users/PCS/Desktop/NEWS Headline Clustering - WOG/complete_cluster2.png', dpi=200)



plt.close()

file1.close()











