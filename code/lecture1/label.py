import cv2
import numpy as np
import os
import pickle
# 对于初步筛选出的selected数据集进行label
path = "../../selected"
files = os.listdir(path)
label=[]
vector = []
cnt = 0
for filename in files:
    add = path+"/"+filename
    img = cv2.imread(add,cv2.IMREAD_GRAYSCALE)
    cv2.imshow(str(cnt),img)
    k=cv2.waitKey(0)
    if k != ord('p'):
        label.append(0)
    else:
        label.append(1)
    cv2.destroyAllWindows()
    vector.append(img)
    cnt+=1
    print(label[cnt-1])
label_output = open('../../selected/label.pkl','wb')
pickle.dump(label,label_output)
label_output.close()
vector_output=open('../../selected/img.pkl','wb')
pickle.dump(vector,vector_output)
vector_output.close()
# 通过灰度值的和设置阈值初步筛选，在手动判断是否为正样本
# cnt = 0
# mark = []
# for i in range(44713):
#     mark.append(0)
# for i in range(44713):
#     if mark[i] == 0:
#         filename = "C:/Users/86133/Desktop/NBA/dir/dir"+str(i)+".png"
#         img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
#         now_sum = img.sum()
#         if now_sum < 350000:
#             cnt+=1
# print(cnt)
#             cv2.imshow(filename,img)
#             k = cv2.waitKey(0)
#             if k != 0:
#                 cv2.destroyAllWindows()
#             if k == ord('p'):
#                 mark[i] = 1
#                 print(i)
#                 print(":\n")
#                 for j in range(10):
#                     now = i+j+1
#                     if mark[now]==0:
#                         filename = "C:/Users/86133/Desktop/NBA/dir/dir" + str(now) + ".png"
#                         img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
#                         cv2.imshow(filename, img)
#                         l = cv2.waitKey(0)
#                         if l != 0:
#                             cv2.destroyAllWindows()
#                         if l == ord('p'):
#                             mark[now] = 1
#                             print(now)
#                             print(" ")
#                     now = i - j - 1
#                     if mark[now] == 0:
#                         filename = "C:/Users/86133/Desktop/NBA/dir/dir" + str(now) + ".png"
#                         img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
#                         cv2.imshow(filename, img)
#                         l = cv2.waitKey(0)
#                         if l != 0:
#                             cv2.destroyAllWindows()
#                         if l == ord('p'):
#                             mark[now] = 1
#                             print(now)
#                             print(" ")
#                 print("\n")
# file_address = "C:/Users/86133/Desktop/NBA"
# data=open("C:/Users/86133/Desktop/NBA/label.txt",'w+')
# print(mark,file=data)