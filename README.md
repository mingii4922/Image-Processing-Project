# Image-Processing-Project

2021_영상처리 프로젝트
---------------------------------
# Introduction

영상처리 프로그래밍을 통해 배운 것들을 활용하여 이미지 처리를 위한 opencv 라이브러리와 얼굴인식을 위한 dlib라이브러리, 
행렬 연산을 위한 numpy를 사용하여 영상에서 사람의 얼굴을 인식하는 프로그램 구현하고 다양한 이벤트를 통해 영상이 바뀌도록 구성하였다.
또, 해당영상에서만 작용하지 않고, 어떠한 영상에서도 구성한 이벤트가 작용할 수 있도록 두개의 영상을 사용하며 코드를 구현하였다.
-------------------------------
# Solution

먼저 영상에서 움직이는 사람의 얼굴의 영역을 검출하기 위해서 사전에 학습되어 있는 ‘shape_predictor_68_face_landmarks.dat’ 얼굴인식 모델을 사용했다.
다음 총 4가지의 이벤트를 만들어 보았는데, 1.blurring, 2.sunglass, 3.red cheek, 4.headband를 만들어 보았다.

각 이벤트들을 함수로 정의하여 사전 학습 모델인 ‘shape_predictor_68_face_landmarks.dat’를 사용해 각각의 이벤트들을 적용할 위치를 68개의 검출된 parameter들을 통해 만들었다.

프로젝트를 진행하면서 검출된 얼굴영역 위에 특별한 위치들만 필요한 sunglass와 headband 이벤트는 bitwise연산을 적용해 좀 더 자연스럽게 만들어보고자 노력했다.

최종 결과로 다양한 영상을 적용하였을 때, 원하는 모션을 추가하여 이미지를 마음대로 처리할 수 있는 프로젝트를 성공적으로 이루었다.

# Headband 처리 영상 결과
ttps://user-images.githubusercontent.com/79297596/126984124-1094c676-1d03-4ddb-8875-5ccc707b2d67.mp4

# Sunglass_face 처리 영상 결과
https://user-images.githubusercontent.com/79297596/126983514-62366447-b787-4b14-a017-da7754f826a0.mp4

------------------------------------------------
# Blurring 처리 영상 결과
https://user-images.githubusercontent.com/79297596/126983799-5469cafc-fc13-4a19-8367-53c8f1f0f346.mp4

# Red Cheek_face 처리 영상 결과

사용한 영상 자료 : https://www.pexels.com/search/videos/face
