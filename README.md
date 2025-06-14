# Brain-Computer-Interfaces

본 프로젝트는 conda 가상환경('environment.yml' 기반)에서 reproduce할 수 있습니다.

---

## Miniconda 설치
* 기존에 conda가 설치되어 있다면 이 과정은 생략해도 됩니다.
### 1) 다운로드
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
### 2) 설치
```bash
bash Miniconda3-latest-Linux-x86_64.sh
```
### 3) 버전 확인
설치 후, 새 터미널에서 아래 명령어로 정상 설치 여부를 확인하세요.

```bash
conda --version
```
## Conda 환경 생성
```bash
conda env create -f environment.yml
```
## 가상 환경 활성화
```bash
conda activate brain_interface
```
## 실험 방법
특정 subject(0-9)에 대한 실험을 실행하고자 할 때는, main.py 파일의 173 line에 있는 subject 값을 원하는 값(0-9)으로 수정한 후 아래 코드 실행 명령어로 실행하면 됩니다.

## 코드 실행
```bash
python3 main.py
```
## 결과 확인
실행 결과는 result.txt에 저장됩니다.

## Github 실행 안내
만약 압축 파일 내 코드로 실행이 되지 않는 경우, 아래의 Github 저장소를 clone하여
이 README의 안내에 따라 코드를 실행해 주세요.

```bash
git clone https://github.com/groundwater98/Brain-Computer-Interfaces.git
cd Brain-Computer-Interfaces
```