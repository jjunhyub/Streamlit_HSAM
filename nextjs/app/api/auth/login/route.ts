// @ts-nocheck

import { NextResponse } from 'next/server';
import { buildSessionCookieValue, sessionCookieOptions } from '../../../../lib/auth/session';
import { verifyReviewerPassword } from '../../../../lib/auth/reviewers';

export const runtime = 'nodejs';

export async function POST(request) {
  let reviewerId = '';
  let password = '';

  const contentType = request.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    const body = await request.json().catch(() => ({}));
    reviewerId = String(body?.reviewerId || '').trim();
    password = String(body?.password || '');
  } else {
    const formData = await request.formData().catch(() => null);
    reviewerId = String(formData?.get('reviewerId') || '').trim();
    password = String(formData?.get('password') || '');
  }

  if (!reviewerId) {
    return NextResponse.json({ error: 'Reviewer ID를 입력하세요.' }, { status: 400 });
  }
  if (!password) {
    return NextResponse.json({ error: '비밀번호를 입력하세요.' }, { status: 400 });
  }
  if (!verifyReviewerPassword(reviewerId, password)) {
    return NextResponse.json({ error: '등록되지 않았거나 비밀번호가 올바르지 않습니다.' }, { status: 401 });
  }

  const response = NextResponse.json({ ok: true });
  response.cookies.set({
    ...sessionCookieOptions,
    value: buildSessionCookieValue(reviewerId),
  });
  return response;
}
