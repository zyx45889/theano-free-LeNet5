from libsvm.python.svmutil import *
from liblinear.python import *
from libsvm.python.svm import *
import matplotlib.pyplot as plt
import pickle
def check_t(c):
    train_label,train_pixel = svm_read_problem('../../selected/traindata.txt')
    practical_label,predict_pixel = svm_read_problem('../../selected/testdata.txt')
    model = svm_train(train_label,train_pixel,'-t '+str(c))
    p_label, p_acc, pp_val = svm_predict(practical_label, predict_pixel, model);
    x=[]
    y=[]
    n=100
    xx = sum(practical_label)
    yy = len(practical_label)-xx
    p_val = []
    for i in range(len(pp_val)):
        p_val.append(pp_val[i][0])
    for i in range(n):
        threshold = (max(p_val)-min(p_val))/n*i+min(p_val)
        print(threshold)
        true_true = 0
        true_flase = 0
        for j in range(len(p_val)):
            if p_val[j]>threshold:
                if practical_label[j]==1:
                    true_true+=1
                else:
                    true_flase+=1
        x.append(true_true/xx)
        y.append(true_flase/yy)
    return x,y

def check_c(c):
    train_label,train_pixel = svm_read_problem('../../selected/traindata.txt')
    practical_label,predict_pixel = svm_read_problem('../../selected/testdata.txt')
    model = svm_train(train_label,train_pixel,'-c '+str(c))
    p_label, p_acc, pp_val = svm_predict(practical_label, predict_pixel, model);
    x=[]
    y=[]
    n=100
    xx = sum(practical_label)
    yy = len(practical_label)-xx
    p_val = []
    for i in range(len(pp_val)):
        p_val.append(pp_val[i][0])
    for i in range(n):
        threshold = (max(p_val)-min(p_val))/n*i+min(p_val)
        print(threshold)
        true_true = 0
        true_flase = 0
        for j in range(len(p_val)):
            if p_val[j]>threshold:
                if practical_label[j]==1:
                    true_true+=1
                else:
                    true_flase+=1
        x.append(true_true/xx)
        y.append(true_flase/yy)
    return x,y

def check_g(c):
    train_label,train_pixel = svm_read_problem('../../selected/traindata.txt')
    practical_label,predict_pixel = svm_read_problem('../../selected/testdata.txt')
    model = svm_train(train_label,train_pixel,'-t 2 -g '+str(c))
    p_label, p_acc, pp_val = svm_predict(practical_label, predict_pixel, model);
    x=[]
    y=[]
    n=100
    xx = sum(practical_label)
    yy = len(practical_label)-xx
    p_val = []
    for i in range(len(pp_val)):
        p_val.append(pp_val[i][0])
    for i in range(n):
        threshold = (max(p_val)-min(p_val))/n*i+min(p_val)
        print(threshold)
        true_true = 0
        true_flase = 0
        for j in range(len(p_val)):
            if p_val[j]>threshold:
                if practical_label[j]==1:
                    true_true+=1
                else:
                    true_flase+=1
        x.append(true_true/xx)
        y.append(true_flase/yy)
    return x,y

# x,y=check_g(1)
# plt.plot(x,y,'ro-',label="g=1")
# x,y=check_g(0.01)
# plt.plot(x,y,label="g=0.01")
# x,y=check_g(0.001)
# plt.plot(x,y,'r-',label="g=0.001")
# plt.xlabel('false alarm number')
# plt.ylabel('missed positive number')
# plt.title('ROC Curve')
# plt.show()

def save_best():
    train_label,train_pixel = svm_read_problem('../../selected/traindata.txt')
    practical_label,predict_pixel = svm_read_problem('../../selected/testdata.txt')
    model = svm_train(train_label,train_pixel,'-c 0.5 -t 2 -g 1')
    p_label, p_acc, pp_val = svm_predict(practical_label, predict_pixel, model);
    x=[]
    y=[]
    n=100
    xx = sum(practical_label)
    yy = len(practical_label)-xx
    p_val = []
    for i in range(len(pp_val)):
        p_val.append(pp_val[i][0])
    for i in range(n):
        threshold = (max(p_val)-min(p_val))/n*i+min(p_val)
        print(threshold)
        true_true = 0
        true_flase = 0
        for j in range(len(p_val)):
            if p_val[j]>threshold:
                if practical_label[j]==1:
                    true_true+=1
                else:
                    true_flase+=1
        x.append(true_true/xx)
        y.append(true_flase/yy)
    plt.plot(x, y, 'r-', label="g=0.001")
    plt.xlabel('false alarm number')
    plt.ylabel('missed positive number')
    plt.title('ROC Curve')
    plt.show()
    # 储存roc曲线方便后面对比
    # roc=[]
    # roc.append(x)
    # roc.append(y)
    # f = open('C:/Users/86133/Desktop/NBA/libsvm/roc.pkl', 'wb')
    # pickle.dump(roc, f)
    # f.close()

save_best()