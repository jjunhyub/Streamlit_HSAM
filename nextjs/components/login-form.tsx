// @ts-nocheck
'use client';

import { useRouter } from 'next/navigation';
import { useState } from 'react';

export default function LoginForm() {
  const router = useRouter();
  const [reviewerId, setReviewerId] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [pending, setPending] = useState(false);

  async function onSubmit(event) {
    event.preventDefault();
    setPending(true);
    setError('');

    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reviewerId, password }),
      });

      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        setError(payload?.error || '로그인에 실패했습니다.');
        return;
      }

      router.replace('/');
      router.refresh();
    } catch (err) {
      setError('로그인 요청 중 오류가 발생했습니다.');
    } finally {
      setPending(false);
    }
  }

  return (
    <form className="loginCard" onSubmit={onSubmit}>
      <div className="loginBadge">Reviewer Access</div>
      <h1 className="loginTitle">Reviewer Login</h1>
      <p className="loginSubtitle">Reviewer ID와 비밀번호를 입력하세요.</p>

      <label className="fieldLabel" htmlFor="reviewer-id">Reviewer ID</label>
      <input
        id="reviewer-id"
        className="textField"
        value={reviewerId}
        onChange={(event) => setReviewerId(event.target.value)}
        placeholder="아이디를 입력하세요."
        autoComplete="username"
      />

      <label className="fieldLabel" htmlFor="reviewer-password">Password</label>
      <input
        id="reviewer-password"
        className="textField"
        type="password"
        value={password}
        onChange={(event) => setPassword(event.target.value)}
        placeholder="비밀번호를 입력하세요."
        autoComplete="current-password"
      />

      {error ? <div className="errorBox">{error}</div> : null}

      <button className="primaryButton" type="submit" disabled={pending}>
        {pending ? '로그인 중...' : '로그인'}
      </button>
    </form>
  );
}
