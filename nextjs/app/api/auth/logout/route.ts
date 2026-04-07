// @ts-nocheck

import { NextResponse } from 'next/server';
import { SESSION_COOKIE_NAME, sessionCookieOptions } from '../../../../lib/auth/session';

export const runtime = 'nodejs';

export async function POST() {
  const response = NextResponse.json({ ok: true });
  response.cookies.set({
    ...sessionCookieOptions,
    name: SESSION_COOKIE_NAME,
    value: '',
    maxAge: 0,
  });
  return response;
}
