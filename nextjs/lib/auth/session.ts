// @ts-nocheck

import crypto from 'node:crypto';
import { cookies } from 'next/headers';
import { base64UrlDecode, base64UrlEncode } from '../utils';

export const SESSION_COOKIE_NAME = 'hsam_review_session';
const SESSION_TTL_MS = 1000 * 60 * 60 * 24 * 7;

function getSessionSecret() {
  const secret = process.env.SESSION_SECRET?.trim();
  if (!secret) {
    throw new Error('SESSION_SECRET is required.');
  }
  return secret;
}

function sign(payload: string) {
  return base64UrlEncode(
    crypto.createHmac('sha256', getSessionSecret()).update(payload).digest(),
  );
}

export function createSessionToken(reviewerId: string) {
  const payload = JSON.stringify({
    reviewerId,
    iat: Date.now(),
    exp: Date.now() + SESSION_TTL_MS,
  });

  return `${base64UrlEncode(payload)}.${sign(payload)}`;
}

export function verifySessionToken(token: string) {
  try {
    const [encodedPayload, encodedSignature] = String(token || '').split('.');
    if (!encodedPayload || !encodedSignature) return null;

    const payload = base64UrlDecode(encodedPayload);
    const expectedSignature = sign(payload);

    const sigA = Buffer.from(encodedSignature);
    const sigB = Buffer.from(expectedSignature);
    if (sigA.length !== sigB.length || !crypto.timingSafeEqual(sigA, sigB)) {
      return null;
    }

    const parsed = JSON.parse(payload);
    if (!parsed?.reviewerId || !parsed?.exp) return null;
    if (Date.now() > Number(parsed.exp)) return null;

    return {
      reviewerId: String(parsed.reviewerId),
      exp: Number(parsed.exp),
    };
  } catch {
    return null;
  }
}

export async function getSession() {
  const cookieStore = await cookies();
  const token = cookieStore.get(SESSION_COOKIE_NAME)?.value;
  if (!token) return null;
  return verifySessionToken(token);
}

export function buildSessionCookieValue(reviewerId: string) {
  return createSessionToken(reviewerId);
}

export const sessionCookieOptions = {
  name: SESSION_COOKIE_NAME,
  httpOnly: true,
  sameSite: 'lax' as const,
  secure: process.env.NODE_ENV === 'production',
  path: '/',
  maxAge: Math.floor(SESSION_TTL_MS / 1000),
};
