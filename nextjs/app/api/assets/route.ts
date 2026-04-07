// @ts-nocheck

import { NextResponse } from 'next/server';
import { getSession } from '../../../lib/auth/session';
import { downloadAssetFromStorage, getAllowedPathsForReviewerImage } from '../../../lib/review-data';
import { guessContentType, toErrorMessage } from '../../../lib/utils';

export const runtime = 'nodejs';

export async function GET(request) {
  const session = await getSession();
  if (!session) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const url = new URL(request.url);
  const imageId = String(url.searchParams.get('imageId') || '').trim();
  const path = String(url.searchParams.get('path') || '').trim();

  if (!imageId || !path) {
    return NextResponse.json({ error: 'imageId and path are required.' }, { status: 400 });
  }

  try {
    const allowedPaths = await getAllowedPathsForReviewerImage(session.reviewerId, imageId);
    if (!allowedPaths || !allowedPaths.has(path)) {
      return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
    }

    const blob = await downloadAssetFromStorage(path);
    const arrayBuffer = await blob.arrayBuffer();

    return new NextResponse(arrayBuffer, {
      status: 200,
      headers: {
        'Content-Type': blob.type || guessContentType(path),
        'Cache-Control': 'private, max-age=60, stale-while-revalidate=300',
      },
    });
  } catch (error) {
    return NextResponse.json({ error: toErrorMessage(error) }, { status: 500 });
  }
}
