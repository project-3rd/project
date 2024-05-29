'''markdown
# Git and GitHub Usage Guide

## 1. Install Git

먼저, Git을 설치해야 합니다. 운영체제에 따라 설치 방법이 다릅니다.

Windows
Git 공식 사이트로 이동합니다.
"Download" 버튼을 클릭하여 최신 버전의 Git을 다운로드합니다.
다운로드한 설치 파일을 실행하고 설치 마법사의 지시에 따라 설치를 완료합니다.

MacOS
터미널을 엽니다.
Homebrew가 설치되어 있다면 brew install git 명령어를 입력하여 설치합니다. Homebrew가 없다면 Homebrew 공식 사이트를 참조하여 Homebrew를 먼저 설치한 후 Git을 설치합니다.

Linux
터미널을 엽니다.
배포판에 따라 다음 명령어 중 하나를 입력하여 Git을 설치합니다:
Ubuntu: sudo apt-get install git
Fedora: sudo dnf install git

## 2. Configure Git

설치가 완료되면 Git을 설정합니다. 사용자 이름과 이메일을 설정합니다.

'''bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"


## 3. Clone Repository

프로젝트 저장소를 클론하여 로컬로 가져옵니다.

'''bash
git clone https://github.com/project-3rd/project.git
cd project



## 4. Create New Branch

작업할 새로운 브랜치를 생성하고 이동합니다.

'''bash
git checkout -b my-feature-branch




## 5. Make Changes



코드를 수정하거나 필요한 작업을 수행합니다.


## 6. Commit Changes


변경 사항을 스테이징하고 커밋합니다.

'''bash
git add .
git commit -m "Add detailed explanation on Git usage"


## 7. Push to Remote


변경 사항을 원격 저장소의 새로운 브랜치에 푸시합니다.

'''bash
git push origin my-feature-branch


## 8. Create Pull Request

GitHub 웹사이트로 이동하여, 방금 푸시한 브랜치에 대해 Pull Request를 생성합니다.

저장소 페이지로 이동합니다.
"Pull requests" 탭을 클릭합니다.
"New pull request" 버튼을 클릭합니다.
"compare"에서 방금 만든 브랜치를 선택하고 "Create pull request" 버튼을 클릭합니다.
제목과 설명을 작성하고 "Create pull request"를 클릭하여 PR을 생성합니다.


## 9. Review and Merge


동료들이 PR을 리뷰하고 승인하면, PR을 병합합니다. 병합이 완료되면 로컬 저장소를 업데이트합니다.

'''bash
git checkout main
git pull origin main
