# Brain-Computer-Interfaces

본 프로젝트는 conda 가상환경('environment.yml' 기반)에서 reproduce할 수 있습니다.

---

## Conda 환경 생성
conda env create -f environment.yml


## 가상 환경 활성화
conda activate brain_interface

## 실험 방법
특정 subject(0-9)에 대한 실험을 실행하고자 할 때는, main.py 파일의 173 line에 있는 subject 값을 원하는 값(0-9)으로 수정한 후 아래 코드 실행 명령어로 실행하면 됩니다.

## 코드 실행
python3 main.py

## 결과 확인
실행 결과는 result.txt에 저장됩니다.

