# H-SAM Review Tool — Next.js migration

Streamlit에서 쓰던 reviewer UI를 Next.js(App Router)로 옮긴 버전입니다.
기존 Supabase 테이블/스토리지를 그대로 쓰는 것을 전제로 만들었고, 외부 배포를 고려해 **DB/Storage 접근은 전부 서버에서만** 처리되도록 구성했습니다.

## 무엇을 그대로 가져왔나

- reviewer 로그인 흐름
- `review_assignments` → `image_manifest` 조회 구조
- `review_annotations` payload 저장 구조
- `review-dataset` bucket의 원본/마스크/인스턴스 이미지 사용
- node question / tree question / completion 계산 로직
- 15초 autosave
- left image list + right inspector 레이아웃
- hierarchy tree + 전체 트리 질문 진입 조건

## 이 버전에서 바꾼 점

- Streamlit session state 대신 React state 사용
- Storage 파일은 브라우저가 Supabase를 직접 치지 않고 `/api/assets`를 통해서만 접근
- reviewer 비밀번호는 코드 파일이 아니라 `REVIEWER_USERS_JSON` 환경변수로 관리
- reviewer 세션은 httpOnly signed cookie로 관리

## 필요한 환경변수

`.env.example` 참고.

필수:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_BUCKET`
- `REVIEWER_USERS_JSON`
- `SESSION_SECRET`

권장:

- `NEXT_PUBLIC_APP_TITLE`
- `APP_TITLE`

## reviewer 비밀번호 해시 만들기

```bash
node scripts/hash-reviewer-passwords.mjs reviewer_a=my-password reviewer_b=another-password
```

출력된 JSON 전체를 `REVIEWER_USERS_JSON`에 넣으면 됩니다.

## 설치 / 로컬 실행

```bash
npm install
npm run dev
```

## Vercel 배포

```bash
vercel
```

배포 후 Vercel Project Settings → Environment Variables 에 아래 값을 넣으면 됩니다.

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_BUCKET`
- `SUPABASE_REVIEW_TABLE`
- `SUPABASE_ASSIGNMENTS_TABLE`
- `SUPABASE_MANIFEST_TABLE`
- `REVIEWER_USERS_JSON`
- `SESSION_SECRET`
- `NEXT_PUBLIC_APP_TITLE`
- `APP_TITLE`

## 전제하는 Supabase 스키마

### 1) `review_assignments`

최소 컬럼:

- `reviewer_id`
- `image_id`
- `sort_index`

### 2) `image_manifest`

최소 컬럼:

- `image_id`
- `root_image_path`
- `root_overlay_path`
- `full_size`
- `nodes`
- `roots`
- `actual_nodes`

### 3) `review_annotations`

최소 컬럼:

- `reviewer_id`
- `payload`
- `updated_at`

`upsert()`를 쓰기 때문에 `reviewer_id`에는 unique index 또는 primary key가 있는 편이 좋습니다.

예시:

```sql
create unique index if not exists review_annotations_reviewer_id_idx
on public.review_annotations (reviewer_id);
```

## Storage 버킷

기본값은 `review-dataset`입니다.

이 프로젝트는 private bucket도 동작하도록 `/api/assets`에서 reviewer의 assignment와 manifest를 확인한 다음 파일을 중계합니다.
즉, 서비스 role key가 브라우저로 내려가지 않습니다.

## 번역 파일

기존 Streamlit의 `full_translated.json`을 쓰고 싶으면 아래 위치에 두면 됩니다.

```text
public/translation/full_translated.json
```

없으면 기본 영어 label만 표시됩니다.

## 보안 메모

- `SUPABASE_SERVICE_ROLE_KEY`는 절대 `NEXT_PUBLIC_` 접두사를 붙이면 안 됩니다.
- `SESSION_SECRET`는 충분히 길고 랜덤해야 합니다.
- 완전한 public internet 서비스로 열 생각이면 로그인 rate limit, audit log, WAF를 추가하는 것이 좋습니다.
- Storage 직접 public 공개 대신 현재처럼 서버 게이트를 두는 편이 안전합니다.

## 남겨둔 개선 포인트

- reviewer를 eventually Supabase Auth로 이관
- asset allow-list를 Redis/Edge cache로 최적화
- 더 정교한 tree virtualization
- autosave 실패 재시도 큐
- reviewer activity log / admin dashboard
