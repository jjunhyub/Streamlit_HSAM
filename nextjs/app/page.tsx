// @ts-nocheck

import { redirect } from 'next/navigation';
import ReviewApp from '../components/review-app';
import { getSession } from '../lib/auth/session';
import { firstReviewableNodeId } from '../lib/review-logic';
import { loadReviewPageData } from '../lib/review-data';

export const dynamic = 'force-dynamic';

export default async function HomePage({ searchParams }) {
  const session = await getSession();
  if (!session) {
    redirect('/login');
  }

  const params = (await searchParams) || {};

  try {
    const { records, annotations } = await loadReviewPageData(session.reviewerId);
    const imageIds = Object.keys(records || {});

    let initialImageId = String(params.image || '').trim();
    if (!initialImageId || !records[initialImageId]) {
      initialImageId = imageIds[0] || null;
    }

    const selectedRecord = initialImageId ? records[initialImageId] : null;
    const requestedMode = String(params.mode || 'node');
    const initialMode = requestedMode === 'tree' ? 'tree' : 'node';

    let initialNodeId = null;
    if (selectedRecord && initialMode === 'node') {
      const requestedNodeId = String(params.node || '').trim();
      if (requestedNodeId && selectedRecord.nodes?.[requestedNodeId]) {
        initialNodeId = requestedNodeId;
      } else {
        initialNodeId = firstReviewableNodeId(selectedRecord);
      }
    }

    return (
      <ReviewApp
        reviewerId={session.reviewerId}
        records={records}
        initialAnnotations={annotations}
        initialSelection={{
          imageId: initialImageId,
          mode: initialMode,
          nodeId: initialNodeId,
        }}
      />
    );
  } catch (error) {
    return (
      <main className="errorPage">
        <div className="errorPageCard">
          <h1>데이터를 불러오지 못했습니다.</h1>
          <p>{error instanceof Error ? error.message : String(error)}</p>
        </div>
      </main>
    );
  }
}
