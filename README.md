# Track2Vec

trac2vec is a library for efficient learning of track, artist, and genre representations.

## Building Track2Vec using cmake
Build system of track2vec only supports Linux and Macos.

```
$ git clone https://github.com/okwon78/track2vec
$ cd track2vec
$ mkdir build && cd build && cmake ..
$ make
```

## Arguments
The followings arguments are requried for training. 
```bash
$ track2vec train <arguments>
```
|Args|discription|default value|
|------|---|---|
| -input| 학습 데이터 (json) | N/A (필수) |
| -output| 결과물을 저장 할 디렉토리 | N/A (필수) |
| -meta | 학습에 필요한 메타 파일 | N/A (필수) |
| -s3log | 학습 로그를 저장할 s3 위치 | N/A (필수) |
| -locallog | 학습 로그를 저장할 local 위치 | N/A (필수) |
| -yyyymmdd | 로그에 사용할 학습 시작 일 | N/A (필수) |
| -lr | 초기 lr (progress에 따라 decay 됨) | 0.1 |
| -pretrained_lr | 학습된 Embedding에 적용될 lr 비율 | 0.2 |
| -dim | Embedding 길이 | 200 |
| -ws | window size | 5 |
| -epoch | epoch | 10 |
| -neg | negative sampling | 10 |
| -seed | random seed | 0 |
| -printInterval | 학습 로그를 생성 주기 (초 단위) | 1 |
| -logBufferSize | 생성된 로그를 s3 올리기 위한 버퍼링 크기 | 0 |
| -lrUpdateRate | 지정된 값 만큼 토큰이 처리될 때 마다 progress에 따라 lr 변경 | 10000 |
| -verbose | 로그 레벨 | 1 |
| -thread | 학습에 사용될 thread 수 | 컴퓨터의 코어 갯수 |
| -threadInterval | 학습 데이터 파일에서 thread 시작 위치 간격 | 200 |
| -discard_t | 각 토큰의 discard rate에 사용되는 상수 값 | 0.0001 |
| -es | early stop 체크 시작 loss | 1.0 |


