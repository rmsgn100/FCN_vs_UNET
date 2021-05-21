# FCN vs UNET 
파일 소개
> 데이터셋 : 2d_images, 2d_masks
> 
> 저장한 모델 : final_model
> 
> 작업을 진행한 코랩파일 : Solo_Project_4.ipy
> 
> 콘다 가상환경을 만들어 재구현한 파일 : copy_project4.ipynb
> 
> 저장한 가중치 : model_weights.h5
> 
> Dependency : requirements.txt

<p align="center">
<img width="800" alt="portfolio_view" src="https://user-images.githubusercontent.com/61172021/104187100-b5463880-545a-11eb-891b-998ffd43a9ac.png">
</p>

# Intro

AI 는 사람이 하는일을 대신 해 줌으로써 우리에게 엄청난 편의를 제공할 수 있는 아주 이로운 기술입니다. 하지만 일각에선 이것이 사회에 미치는 부정적인 영향에 대한 논란이 끊이지 않습니다. 새로운 산업이 생겨나면 동시에 퇴보하는 산업이 있기 마련인데요, 예를들어 자동차가 생겨나면서 인력거라는 직업이 없어지는 것처럼, AI 가 어떤이들의 일자리를 뺏기도 할 것 같습니다. 그렇기때문에 저는 AI 가 상업적인 방향으로 뿐만이 아니라 사회에 선영향을 미칠 수 있는 영역을 넓히는 것 또한 중요하다고 생각합니다. 대표적으로 의료분야에서의 AI 가 있습니다. AI 의 정확도 개선은 어느 분야에서나 중요하겠지만, 의료분야에서는 AI 의 정확도가 많은 사람들의 생명과 관련이 있기때문에 그 의미가 더 크게 느껴집니다.

<img align="right" width="400" height="410" src="https://user-images.githubusercontent.com/61172021/104192882-cd21ba80-5462-11eb-8916-e0091a95b73d.png">

CT 나 MRI 같은 촬영을 통한 검사 결과를 몇주동안 기다려야하는 경우가 많은데요, 그 이유는 의사가 한명의 의료 영상을 분석하기 위해선 많은 시간(보통 30분 정도)이 필요하기 때문입니다. 의사들의 시간은 환자들의 목숨과도 직결되기 때문에 정말 소중한데요, 요즘은 AI 를 사용해서 의사들이 병진단을 할 때 도움을 줄 수 있는 방법이 다양해 졌다고 합니다. 실제로 X-ray, MRI, CT 같은 의료 영상 혹은 이미지를 분석하는 AI 가 활발히 개발되고있어 의사들의 진단을 크게 돕고있다고 합니다. 
제가 이 프로젝트에서 다루는 것은 폐 CT 촬영 이미지에 관한 것입니다. 이런 분석의 첫단계가 폐 영역을 마스킹하는 과정인데요, 전 이것을 Semantic Segmentation 을 사용해서 자동화할 수 있지않을까? 라는 궁금증이 들었고, Semantic Segmentation 을 위해 사용되는 FCN 과 UNET 이라는 모델들의 성능을 비교해 보았습니다. 이때 저는 UNET 이 FCN 을 발전시킨 모델이기때문에 이것이 더 좋은 성능을 보여줄 것이라는 가설을 세웠습니다. 

- 풀고자 하는 문제 : CT 사진에서 폐영역을 마스킹하는 작업을 Semantic Segmentation 을 사용해 자동화 할 수 있을까?
- 검증하고자 하는 가설 : FCN 보다 그것을 발전시킨 모델인 UNET 이 이 문제를 해결하기위한 더 적합한 모델일 것이다. 

<br>

# 데이터

<img align="right" width="300" height="300" src="https://user-images.githubusercontent.com/61172021/104196565-47543e00-5467-11eb-95de-794559037a6a.png">

<br/>
<br/>

학습과 평가에 사용한 데이터는 CT 로 폐를 촬영한 사진입니다. 원래 CT 촬영을하면 3D 이미지가 결과로 나오는데 이것을 분석하기 쉽게 2D 형식으로 슬라이싱한 사진들입니다. 왼쪽에 있는 사진들이 모델학습에서 X 값이 되는 본래의 CT 사진이고, 오른쪽이 y 값이되는 사람이 직접 폐영역을 마스킹한 사진입니다. 

- X (원래 CT 사진) : 267 개
- y (마스킹된 사진) : 267 개

원래 이미지의 크기는 512 x 512 인데, 이대로 사용하면 콜랩에서 Out of Memory 에러가 발생해서 256 x 256 으로 줄여서 사용했습니다. 

<br/>
<br/>


# FCN 

<img width="1324" alt="스크린샷 2021-01-12 오전 12 29 01" src="https://user-images.githubusercontent.com/61172021/104201504-31497c00-546d-11eb-80a8-cf9870b6d86b.png">

비교해 볼 첫번째 모델로 FCN 을 보시겠습니다. FCN 은 Fully Convolutional Network 의 약자이고, 이것의 목적은 Semantic Segmentation 입니다. Semantic Segmentation 이란, 위의 그림과같이 이미지의 각 픽셀이 어느 클래스에 속하는지 나타내는 것입니다. FCN 은 CNN 기반의 모델이고, Encoder 파트와 Decoder 파트로 구성됩니다. 그리고, 기존의 CNN 기반 모델들은 출력층에 Fully Connected Layer 를 사용하는데, 그러면 이미지의 위치정보를 잃기때문에 FCN 에서는 Convolution Layer 로 대체하는 특징을 가지고있습니다. 그러한 특징때문에 Fully Convolutional Network 라고 이름이 지어졌습니다!

# UNET

<p align="center">
<img width="480" alt="스크린샷 2021-01-12 오전 1 34 38" src="https://user-images.githubusercontent.com/61172021/104210537-5989a880-5476-11eb-9ba5-b6e14f4a0ee0.png">
</p>

UNET 은 FCN 을 발전시킨 모델입니다. FCN 과 마찬가지로 인코딩과 디코딩 파트로 구성되어 있다고 보시면 되는데, 그것들을 UNET 에선 Contracting Path 와 Expanding Path 라고 부릅니다. Contracting Path 에서는 원본 이미지의 특징을 함축하는 과정이 이루어지고, Expanding Path 에서는 함축된 피쳐맵을 확장하는 과정이 이루어집니다. Expanding Path 의 업샘플링하는 과정에서 Concatenate 를 사용하는 것이 특징인데요, UNET 을 발표한 논문에 따르면 이것을 사용해서 대칭되는 Contracting Path 단계에서 얻어진 피쳐맵과 합치는 과정을 통해 예측 결과의 해상도를 높여준다고 합니다. 


# 좀 더 나은 학습을 위한 작업들

<img width="865" alt="스크린샷 2021-01-12 오전 1 46 19" src="https://user-images.githubusercontent.com/61172021/104211902-fac52e80-5477-11eb-8df2-8ae8fcb83361.png">


의료이미지 특성상 많은 데이터를 확보하기가 힘들어서 모델 학습시 훈련 데이터가 모자라 성능 평가의 신뢰성이 떨어질 수 있어서 다음과 같은 방법으로 데이터를 늘리고, 일반화하는 과정을 거쳤습니다. 

1. Image Augmentation : 이미지 rotation/shift 와 같은 랜덤한 변화를 주어 훈련세트의 다양성을 증가시킴.

2. Cross Validation : 훈련, 평가시 사용되는 데이터의 편중을 줄일 수 있고, 조금 더 일반화된 모델을 만들 수 있다. 모든 데이터셋을 훈련에 사용할 수 있다는 장점도 있음.

# 결과

<p align="center">
<img width="800" alt="스크린샷 2021-01-12 오전 2 03 01" src="https://user-images.githubusercontent.com/61172021/104214032-51336c80-547a-11eb-96a1-e5b372077d73.png">
</p>


위 사진은 두 모델의 예측결과를 시각화 한 것이고, 왼쪽이 FCN, 오른쪽이 UNET 의 결과입니다. 보시다시피 FCN 과 UNET 은 비슷한 정확도를 보였지만, 그 결과이미지를 보시면 해상도에 꽤 큰 차이를 보입니다. 위에서 UNET 을 설명할때 언급했던 Concatenate 을 적용하는 부분 말고는 두 모델을 거의 똑같이 만들었는데 이렇게 차이가 난다는 것을 보고 신기하기도하고 놀랬습니다. 이렇게 제가 세웠던 가설인 'UNET 이 폐 CT 사진을 Semantic Segmentation 하는데에 FCN 보다 더 좋은 성능을 낼것이다' 를 검증해 봤습니다. 그 둘은 정확도에선 큰 차이를 보이진 않았지만 예측값으로 나온 이미지의 해상도에서 큰 차이가 있었습니다. 

# 마치며...

CT 사진에서 분석을 위한 장기기관을 마스킹하는 작업을 FCN 과 UNET 같은 Semantic Segmentation 기술을 사용해 자동화 할 수 있다는 것을 알게되었습니다. 제가 이번 프로젝트에서 진행한 Semantic Segmentation 은 더 나아가 심장 비대증, 뇌 질환, 성장 질환, 안구 질환 등 다양한 의료분야에서 사용될 수 있습니다. 사람이 수동적으로 해야하는 일을 기계가 대신 함으로써 분석 시간을 절약할 수 있고, 그로인해 의사가 환자를 케어할 수 있는 시간이 늘어나 더 많은 환자를 치료할 수 있는 좋은 영향을 가지고 올 수 있다고 생각합니다.

