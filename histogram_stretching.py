import numpy as np, cv2

def add_array(fl_image,array):
    for i in range(0,len(fl_image)):    #hist 배열 중복값 생성
        idx = int(fl_image[i]/gap)
        array[idx] += 1
    return array

def draw_histo(hist, shape=(200,256)):
    full_set = np.full(shape, 255, np.uint8)
    cv2.normalize(hist, hist, 0, shape[0], cv2.NORM_MINMAX)

    for i,h in enumerate(hist):
        x = int(round(i*gap))       #round반올림함수
        w = int(round(gap))
        cv2.rectangle(full_set, (x, 0, w, int(h)), 0, cv2.FILLED)

    return cv2.flip(full_set, 0)

image = cv2.imread("../images/hist_stretch.jpg", cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("영상파일 읽기 오류")

histSize=32
gap = 256/histSize

fl_image1 = image.flatten()

# max=182, min=53
max = np.max(fl_image1)
min = np.min(fl_image1)

ch_image = image.copy()

for i in range(ch_image.shape[0]):
    for j in range(ch_image.shape[1]):
        # k = ch_image.item(i, j)
        # ch_value = (k-min)/(max-min)*255
        # ch_image.itemset((i, j), ch_value)
        ch_image[i,j] = (ch_image[i,j]-min)/(max-min)*255
fl_image2 = ch_image.flatten()


zero1 = np.zeros(histSize, np.float32)
zero2 = np.zeros(histSize, np.float32)

array1 = add_array(fl_image1,zero1)      #중복횟수담은 배열
array2 = add_array(fl_image2,zero2)

hist_img = draw_histo(array1)           #histogram표현이미지
ch_hist_img = draw_histo(array2)

cv2.imshow("image", image)
cv2.imshow("hist_img", hist_img)

cv2.imshow("ch_image", ch_image)
cv2.imshow("ch_hist_img", ch_hist_img)
cv2.waitKey(0)