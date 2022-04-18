import cv2
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt
def HoG(img):
    #img = np.sqrt(img / float(np.max(img)))
    cell_size = 8
    bin_size = 9
    angle_unit = 360 / bin_size
    height, width = img.shape

    width = int(np.ceil(width/cell_size)*cell_size)
    height = int(np.ceil(height/cell_size)*cell_size)
    img = cv2.resize(img,(width,height),interpolation=cv2.INTER_CUBIC)

    g_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    g_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    g_magnitude = np.sqrt(np.square(g_values_x)+np.square(g_values_y))
    g_angle = cv2.phase(g_values_x, g_values_y, angleInDegrees=True)

    def get_closest_bins(gradient_angle):
        idx = int(gradient_angle / angle_unit) % bin_size
        mod = gradient_angle % angle_unit
        return idx, (idx + 1) % bin_size, mod

    def cell_gradient(cell_magnitude, cell_angle):
        ret = [0] * bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_magnitude = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = get_closest_bins(gradient_angle)
                ret[min_angle] += (gradient_magnitude * (1 - (mod / angle_unit)))
                ret[max_angle] += (gradient_magnitude * (mod / angle_unit))
        return ret

    # def render_gradient(image, cell_histogram):
    #     max_magnitude = np.array(cell_histogram).max()
    #     for x_height in range(cell_histogram.shape[0]):
    #         for y_width in range(cell_histogram.shape[1]):
    #             cell_hist = cell_histogram[x_height][y_width]
    #             cell_hist /= max_magnitude
    #             x_center = 5*(x_height)*cell_size+int(cell_size*5)/2
    #             y_center = 5*(y_width)*cell_size+int(cell_size*5)/2
    #             angle = 0
    #             for magnitude in cell_hist:
    #                 angle_radian = math.radians(angle)
    #                 cell_width = int(cell_size) / 2 * 5
    #                 x1 = int(x_center + cell_width*5 * math.cos(angle_radian))
    #                 y1 = int(y_center - cell_width*5 * math.sin(angle_radian))
    #                 x2 = int(x_center - cell_width*5 * math.cos(angle_radian))
    #                 y2 = int(y_center + cell_width*5 * math.sin(angle_radian))
    #                 cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
    #                 angle += angle_unit
    #     return image

    cell_histogram_vector = np.zeros((int(height / cell_size),int(width / cell_size), bin_size))
    for i in range(cell_histogram_vector.shape[0]):
        for j in range(cell_histogram_vector.shape[1]):
            cell_magnitude = g_magnitude[i * cell_size:(i + 1) * cell_size,j * cell_size:(j + 1) * cell_size]
            cell_angle = g_angle[i * cell_size:(i + 1) * cell_size,j * cell_size:(j + 1) * cell_size]
            cell_histogram_vector[i][j] = cell_gradient(cell_magnitude, cell_angle)

    # hog_image = render_gradient(np.zeros([height*5, width*5]), cell_histogram_vector)
    hog_vector = []
    for i in range(cell_histogram_vector.shape[0] - 1):
        for j in range(cell_histogram_vector.shape[1] - 1):
            block_vector = []
            block_vector.extend(cell_histogram_vector[i][j])
            block_vector.extend(cell_histogram_vector[i][j + 1])
            block_vector.extend(cell_histogram_vector[i + 1][j])
            block_vector.extend(cell_histogram_vector[i + 1][j + 1])
            mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
            magnitude = mag(block_vector)
            if magnitude != 0:
                normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                block_vector = normalize(block_vector, magnitude)
            hog_vector.extend(block_vector)
    #plt.imshow(hog_image, cmap=plt.cm.gray)
    #plt.show()
    #cv2.imwrite('C:/Users/86133/Desktop/NBA/testHoG.png', hog_image)
    return hog_vector

pkl_file = open('../../selected/img.pkl', 'rb')
data1 = pickle.load(pkl_file)
dir_num = len(data1)
print(dir_num)
print(data1[0])
vector = []
for i in range(dir_num):
    now_vector = HoG(data1[i])
    vector.append(now_vector)
hog_output = open('../../selected/hog.pkl','wb')
pickle.dump(vector,hog_output)
hog_output.close()
label = []
# 利用hog向量间距离设置阈值画roc曲线，在重新设置数据为selected部分之后已不可用
'''
with open("../../label.txt", "r") as f:
    data = f.read()
middle = data.split()
for i in range(len(middle)):
    if tag[i]==0:
        continue
    if middle[i]=="1,":
        label.append(1)
    else:
        label.append(0)
label_output = open('../../label.pkl','wb')
pickle.dump(label,label_output)
label_output.close()
postive_vector = [0]*len(vector[0])
cnt = 0
for i in range(int(0.7*dir_num)):
    if label[i] == 1:
        postive_vector = np.array(postive_vector)+np.array(vector[i])
        cnt+=1
postive_vector=[x/cnt for x in postive_vector]

roc_x = []
roc_y = []
distance = []
cnt_positive = 0
cnt_negative = 0
for j in range(dir_num-int(0.7*dir_num)):
    i = j+int(0.7*dir_num)
    temp1 = np.array(postive_vector)
    temp2 = np.array(vector[i])
    distance.append(np.linalg.norm(temp1 - temp2))
    if label[i] == 1:
        cnt_positive += 1
    else:
        cnt_negative += 1

test_num = 200
for i in range(test_num):
    threshold = max(distance)/test_num*i
    nowx = 0
    nowy = 0
    for j in range(dir_num-int(0.7*dir_num)):
        now=j+int(0.7*dir_num)
        if distance[j] < threshold:
            if label[now] == 1:
                nowx += 1
            else:
                nowy += 1
    nowx /= cnt_positive
    nowy /= cnt_negative
    roc_x.append(nowx)
    roc_y.append(nowy)

x=np.array(roc_x)
y=np.array(roc_y)
plt.scatter(x,y,s=5)
plt.xlabel('false alarm number')
plt.ylabel('missed positive number')
plt.title('ROC Curve')
plt.show()
'''