import pickle

hog_file = open('../../selected/hog.pkl', 'rb')
data_x = pickle.load(hog_file)
label_file = open('../../selected/label.pkl', 'rb')
data_y = pickle.load(label_file)
file_name1="../../selected/traindata.txt"
file_name2="../../selected/testdata.txt"
f1=open(file_name1,"w")
f2=open(file_name2,"w")
l=len(data_x)
for i in range(int(l*0.6)):
    print(data_y[i],file=f1,end=' ')
    for j in range(len(data_x[i])):
        if j!=len(data_x[i])-1 :
            print(j+1,":",data_x[i][j]," ",file=f1,end='',sep='')
        else:
            print(j + 1, ":", data_x[i][j],"\n",file=f1, end='', sep='')
for x in range(l-int(l*0.6)):
    i=(int(l*0.6))+x
    print(data_y[i],file=f2,end=' ')
    for j in range(len(data_x[i])):
        if j != len(data_x[i]) - 1:
            print(j + 1, ":", data_x[i][j], " ", file=f2, end='', sep='')
        else:
            print(j + 1, ":", data_x[i][j], "\n", file=f2, end='', sep='')