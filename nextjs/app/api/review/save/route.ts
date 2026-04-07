// @ts-nocheck

import { NextResponse } from 'next/server';
import { getSession } from '../../../../lib/auth/session';
import { saveAnnotationsForReviewer } from '../../../../lib/review-data';
import { nowIso, toErrorMessage } from '../../../../lib/utils';

export const runtime = 'nodejs';

export async function POST(request) {
  const session = await getSession();
  if (!session) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const body = await request.json().catch(() => null);
  if (!body || typeof body !== 'object' || typeof body.annotations !== 'object') {
    return NextResponse.json({ error: 'Invalid payload' }, { status: 400 });
  }

  try {
    await saveAnnotationsForReviewer(session.reviewerId, body.annotations);
    return NextResponse.json({ ok: true, savedAt: nowIso() });
  } catch (error) {
    return NextResponse.json({ error: toErrorMessage(error) }, { status: 500 });
  }
}
