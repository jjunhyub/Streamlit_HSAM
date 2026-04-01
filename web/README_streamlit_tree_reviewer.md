# Streamlit Tree Reviewer 초안

실행:

```bash
pip install -r requirements_streamlit_tree_reviewer.txt
streamlit run streamlit_tree_reviewer_app.py
```

## 기본 동작
- 왼쪽: image id 리스트 (scrollable)
- 가운데: tree node 리스트 (scrollable)
- 오른쪽: 선택한 node의 이미지 + 질문
- 모든 node 완료 후에만 `전체 트리 질문` 버튼 활성화
- 전체 트리 질문까지 끝나면 왼쪽 image 박스가 파란색으로 표시
- 완료되지 않은 상태에서 완료 확인을 누르면 놓친 node / 질문 id를 표시

## 실제 데이터 연결
사이드바의 `Dataset root` 에 아래처럼 상위 폴더를 넣으면 됩니다.

```text
C:\dataset_root
├─000000010114
├─000000010115
├─000000010123
└─000000010125
```

각 image 폴더 내부의 1-depth 하위 폴더 이름을 읽어서 tree를 구성합니다.
`__subcrops` 폴더는 node로 취급하지 않고, 선택한 node의 시각자료 탐색용으로만 사용합니다.

## 나중에 바꾸면 좋은 부분
- `node_questions_for()` : 노드 질문 문항 정의
- `tree_questions_for()` : 전체 트리 질문 정의
- `node_assets()` : 파일명 규칙별 원본 / overlay / b01 분류 규칙
- `persist_annotations()` : JSON 대신 SQLite/Postgres로 교체 가능
