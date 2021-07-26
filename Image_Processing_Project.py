import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture('video.mp4')
# cap = cv2.VideoCapture('walking_man.mp4')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#볼찍기
def cheek(imgorg):
    img_cheek = imgorg.copy()
    for face in faces:
        landmarks = predictor(imgorg, face)
        #왼볼
        x1,x2 = landmarks.part(36).x, landmarks.part(48).x
        y1,y2 = landmarks.part(36).y, landmarks.part(48).y
        x3 = landmarks.part(31).x
        cen_x, cen_y = int((x1+x2)/2), int((y1+y2)/2)
        size = int((x3-cen_x)/2)
        cv2.circle(img_cheek,(cen_x,cen_y),size,(153,70,252),cv2.FILLED)
        #오른볼
        x1,x2 = landmarks.part(45).x, landmarks.part(54).x
        y1,y2 = landmarks.part(45).y, landmarks.part(54).y
        right_x, right_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(img_cheek, (right_x, right_y), size, (153, 70, 252), cv2.FILLED)
    return img_cheek

def sunglasses(imgorg,sunglass_img):
    img_sun = imgorg.copy()
    point0 = (int(sunglass_img.shape[1] * 0.2), int(sunglass_img.shape[0] * 0.5))
    point2 = (int(sunglass_img.shape[1] * 0.8), int(sunglass_img.shape[0] * 0.5))
    Src = [(point0[0], point0[1]), (point2[0], point2[1])] #선글라스 이미지의 기준점

    sunglass_gray = cv2.cvtColor(sunglass_img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(sunglass_gray, 170, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.merge((mask, mask, mask))

    mask_inv = cv2.bitwise_not(mask)
    sunglass_mask = cv2.bitwise_and(sunglass_img, mask)  # 선글라스
    eye_mask = cv2.bitwise_and(sunglass_img, mask_inv)  # 안경마스크와 눈
    eye_mask = cv2.bitwise_not(eye_mask)
    for face in faces:
        landmarks = predictor(imgorg, face)
        left_eye_x, left_eye_y = landmarks.part(36).x, landmarks.part(36).y
        right_eye_x, right_eye_y = landmarks.part(45).x, landmarks.part(45).y
        # 선글라스 이미지의 대응점 저장
        Dst = [(left_eye_x, left_eye_y), (right_eye_x, right_eye_y)]

        # 2D변환 행렬 생성
        ret = cv2.estimateAffinePartial2D(np.array([Src]), np.array([Dst]))
        transform_matrix = ret[0]
        # 선글라스 위치 이동 변환
        transform_sunglass = cv2.warpAffine(sunglass_mask, transform_matrix, (imgorg.shape[1], img_sun.shape[0]))
        # 눈 마스크 이미지 위치 크기 조정
        transform_eye = cv2.warpAffine(eye_mask, transform_matrix, (img_sun.shape[1], img_sun.shape[0]))

        sun_face = cv2.bitwise_and(img_sun, transform_eye)
        sun_face = cv2.addWeighted(sun_face, 0.5, transform_sunglass, 0.4, 0)
        face_without_eye = cv2.bitwise_and(img_sun, cv2.bitwise_not(transform_eye))

        img_sun = cv2.add(sun_face, face_without_eye)
    return img_sun

def head(imgorg,hat_img):
    img_head = imgorg.copy()
    point0 = (int(hat_img.shape[1] * 0.1), int(hat_img.shape[0] * 0.5)) #머리띠 이미지의 기준점
    point2 = (int(hat_img.shape[1] * 0.9), int(hat_img.shape[0] * 0.5))
    Src = [(point0[0]-40, point0[1]), (point2[0], point2[1])]

    head_gray = cv2.cvtColor(hat_img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(head_gray, 180, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.merge((mask, mask, mask))
    mask_inv = cv2.bitwise_not(mask)
    hat_mask = cv2.bitwise_and(hat_img, mask)  # 머리띠
    head_mask = cv2.bitwise_and(hat_img, mask_inv)  # 머리띠마스크
    head_mask = cv2.bitwise_not(head_mask)

    for face in faces:
        # 얼굴영역 좌표
        landmarks = predictor(imgorg, face)
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        y3 = int(y1 + ((y1 - y2) * 0.3))
        # 선글라스 이미지의 대응점 저장
        Dst = [(x1 - 40, y3), (x2 + 10, y3)]

        # 2D변환 행렬 생성
        ret = cv2.estimateAffinePartial2D(np.array([Src]), np.array([Dst]))
        transform_matrix = ret[0]
        # 머리띠 위치 이동 변환
        transform_hat = cv2.warpAffine(hat_mask, transform_matrix, (imgorg.shape[1], img_head.shape[0]))
        # 눈 검출 마스크 이미지 위치 크기 조정
        transform_head = cv2.warpAffine(head_mask, transform_matrix, (img_head.shape[1], img_head.shape[0]))

        head_face = cv2.bitwise_and(img_head, transform_head)
        head_face = cv2.addWeighted(head_face, 0.0, transform_hat, 1.0, 0)
        face_without_eye = cv2.bitwise_and(img_head, cv2.bitwise_not(transform_head))

        img_hat = cv2.add(head_face, face_without_eye)
    return img_hat

def faceblur(imgorg):
    # 얼굴 영역을 검출해 얼굴영역 테두리를 표시하고 그 부분만 blur 처리합니다
    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(imgorg, (x1, y1), (x2, y2), (0, 0, 0), 2)
        b_face = imgorg[y1:y2, x1:x2]
        b_face = cv2.blur(b_face, (10, 10))
        imgorg[y1:y2, x1:x2] = b_face
    return imgorg

while True:
    ret, img = cap.read()
    if not ret:
        break
    hat_img = cv2.imread("./images/headpin.png") #머리띠 이미지
    sunglass_img = cv2.imread("./images/sunglass.png") #선글라스 이미지

    scaler = 0.3 # 영상 사이즈조절
    img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
    imgorg = img.copy()

    faces = detector(imgorg)
    #함수호출
    cheek_face = cheek(imgorg)
    sunglass_face = sunglasses(imgorg,sunglass_img)
    hat_face = head(imgorg, hat_img)
    blur_face = faceblur(imgorg)

    # cv2.imshow("img", img)
    cv2.imshow("cheek_face", cheek_face)
    cv2.imshow("sunglass_face", sunglass_face)
    cv2.imshow("hat_face", hat_face)
    cv2.imshow("blur_img",blur_face)

    cv2.waitKey(1)