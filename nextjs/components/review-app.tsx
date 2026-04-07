// @ts-nocheck
'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import HierarchyTree from './hierarchy-tree';
import ImageList from './image-list';
import QuestionPanel from './question-panel';
import VisualsPanel from './visuals-panel';
import {
  applyAnswerChange,
  firstReviewableNodeId,
  humanLabel,
  imageComplete,
  missingReport,
  nodeProgress,
} from '../lib/review-logic';
import { parseTranslationPayload } from '../lib/translation';

function downloadJson(filename, payload) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

export default function ReviewApp({ reviewerId, records, initialAnnotations, initialSelection }) {
  const [annotations, setAnnotations] = useState(initialAnnotations || {});
  const [selectedImageId, setSelectedImageId] = useState(initialSelection?.imageId || Object.keys(records || {})[0] || null);
  const [selectedMode, setSelectedMode] = useState(initialSelection?.mode || 'node');
  const [selectedNodeId, setSelectedNodeId] = useState(initialSelection?.nodeId || null);
  const [imageSearch, setImageSearch] = useState('');
  const [translationMap, setTranslationMap] = useState({});
  const [saveStatus, setSaveStatus] = useState({ status: 'idle', savedAt: null, message: '' });

  const annotationsRef = useRef(annotations);
  const dirtyRef = useRef(false);
  const savingRef = useRef(false);

  useEffect(() => {
    annotationsRef.current = annotations;
  }, [annotations]);

  useEffect(() => {
    let cancelled = false;
    fetch('/translation/full_translated.json')
      .then((response) => (response.ok ? response.json() : null))
      .then((payload) => {
        if (!cancelled && payload) {
          setTranslationMap(parseTranslationPayload(payload));
        }
      })
      .catch(() => {
        // Optional file. Ignore 404 or parse failures.
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const imageIds = Object.keys(records || {});
  const selectedRecord = selectedImageId ? records[selectedImageId] : null;

  useEffect(() => {
    if (!imageIds.length) return;
    if (!selectedImageId || !records[selectedImageId]) {
      const nextImageId = imageIds[0];
      setSelectedImageId(nextImageId);
      setSelectedMode('node');
      setSelectedNodeId(firstReviewableNodeId(records[nextImageId]));
    }
  }, [imageIds, records, selectedImageId]);

  useEffect(() => {
    if (!selectedRecord) return;
    if (selectedMode === 'tree') {
      setSelectedNodeId(null);
      return;
    }

    if (!selectedNodeId || !selectedRecord.nodes?.[selectedNodeId]) {
      setSelectedNodeId(firstReviewableNodeId(selectedRecord));
    }
  }, [selectedImageId, selectedMode, selectedNodeId, selectedRecord]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (selectedImageId) params.set('image', selectedImageId);
    if (selectedMode) params.set('mode', selectedMode);
    if (selectedMode === 'node' && selectedNodeId) params.set('node', selectedNodeId);
    else params.delete('node');
    window.history.replaceState(null, '', `/?${params.toString()}`);
  }, [selectedImageId, selectedMode, selectedNodeId]);

  const saveNow = useCallback(async (force = false) => {
    if (savingRef.current) return false;
    if (!force && !dirtyRef.current) return true;

    savingRef.current = true;
    if (!force) {
      setSaveStatus((prev) => ({ ...prev, status: 'saving', message: '저장 중...' }));
    }

    try {
      const response = await fetch('/api/review/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ annotations: annotationsRef.current }),
      });

      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload?.error || '저장에 실패했습니다.');
      }

      dirtyRef.current = false;
      setSaveStatus({
        status: 'saved',
        savedAt: payload?.savedAt || new Date().toISOString(),
        message: '클라우드에 저장됨',
      });
      return true;
    } catch (error) {
      setSaveStatus({
        status: 'error',
        savedAt: null,
        message: error instanceof Error ? error.message : '저장에 실패했습니다.',
      });
      return false;
    } finally {
      savingRef.current = false;
    }
  }, []);

  useEffect(() => {
    const interval = window.setInterval(() => {
      if (dirtyRef.current) {
        void saveNow(false);
      }
    }, 15000);

    return () => window.clearInterval(interval);
  }, [saveNow]);

  useEffect(() => {
    const onBeforeUnload = () => {
      if (!dirtyRef.current) return;
      const blob = new Blob([JSON.stringify({ annotations: annotationsRef.current })], {
        type: 'application/json',
      });
      navigator.sendBeacon('/api/review/save', blob);
    };

    window.addEventListener('beforeunload', onBeforeUnload);
    return () => window.removeEventListener('beforeunload', onBeforeUnload);
  }, []);

  const handleAnswerChange = useCallback((mode, questionId, value, nodeId = null) => {
    if (!selectedImageId) return;
    setAnnotations((prev) => applyAnswerChange(prev, selectedImageId, mode, questionId, value, nodeId));
    dirtyRef.current = true;
    setSaveStatus({ status: 'dirty', savedAt: null, message: '저장되지 않은 변경사항' });
  }, [selectedImageId]);

  const handleSelectImage = useCallback((imageId) => {
    const record = records[imageId];
    setSelectedImageId(imageId);
    setSelectedMode('node');
    setSelectedNodeId(firstReviewableNodeId(record));
  }, [records]);

  const handleSelectNode = useCallback((nodeId) => {
    setSelectedMode('node');
    setSelectedNodeId(nodeId);
  }, []);

  const handleSelectTreeSummary = useCallback(() => {
    setSelectedMode('tree');
    setSelectedNodeId(null);
  }, []);

  const handleLogout = useCallback(async () => {
    await saveNow(true);
    await fetch('/api/auth/logout', { method: 'POST' }).catch(() => null);
    window.location.href = '/login';
  }, [saveNow]);

  const selectedMissingReport = useMemo(() => {
    if (!selectedRecord || !selectedImageId) return null;
    return missingReport(annotations, selectedImageId, selectedRecord);
  }, [annotations, selectedImageId, selectedRecord]);

  const selectedProgress = useMemo(() => {
    if (!selectedRecord || !selectedImageId) return [0, 0];
    return nodeProgress(annotations, selectedImageId, selectedRecord);
  }, [annotations, selectedImageId, selectedRecord]);

  if (!imageIds.length) {
    return (
      <main className="appShell">
        <header className="topBar">
          <div>
            <div className="topBarTitle">{process.env.NEXT_PUBLIC_APP_TITLE || 'H-SAM Review Tool'}</div>
            <div className="topBarMeta">Reviewer ID: {reviewerId}</div>
          </div>
          <div className="topBarActions">
            <button className="secondaryButton" onClick={handleLogout}>로그아웃</button>
          </div>
        </header>

        <div className="emptyStateCard">
          <h1>할당된 이미지가 없습니다.</h1>
          <p>Reviewer <strong>{reviewerId}</strong> 에게 연결된 review_assignments 행이 없습니다.</p>
        </div>
      </main>
    );
  }

  const [done, total] = selectedProgress;
  const completed = selectedRecord && selectedImageId ? imageComplete(annotations, selectedImageId, selectedRecord) : false;

  return (
    <main className="appShell">
      <header className="topBar">
        <div>
          <div className="topBarTitle">{process.env.NEXT_PUBLIC_APP_TITLE || 'H-SAM Review Tool'}</div>
          <div className="topBarMeta">Reviewer ID: {reviewerId}</div>
        </div>

        <div className="topBarActions">
          <div className={`saveBadge is-${saveStatus.status}`}>
            {saveStatus.message || '대기 중'}
            {saveStatus.savedAt ? ` · ${new Date(saveStatus.savedAt).toLocaleString('ko-KR')}` : ''}
          </div>
          <button className="secondaryButton" onClick={() => saveNow(true)}>☁️ 클라우드에 저장</button>
          <button
            className="secondaryButton"
            onClick={() => downloadJson(`annotations_${reviewerId}.json`, annotations)}
          >
            현재 annotation JSON 다운로드
          </button>
          <button className="secondaryButton" onClick={handleLogout}>로그아웃</button>
        </div>
      </header>

      <div className="mainGrid">
        <ImageList
          records={records}
          annotations={annotations}
          selectedImageId={selectedImageId}
          imageSearch={imageSearch}
          onImageSearch={setImageSearch}
          onSelectImage={handleSelectImage}
        />

        <section className="contentColumn">
          {selectedRecord ? (
            <>

              <HierarchyTree
                record={selectedRecord}
                annotations={annotations}
                selectedMode={selectedMode}
                selectedNodeId={selectedNodeId}
                onSelectNode={handleSelectNode}
                onSelectTreeSummary={handleSelectTreeSummary}
                translationMap={translationMap}
              />

              {selectedMode === 'node' && selectedNodeId ? (
                <>
                  <VisualsPanel
                    record={selectedRecord}
                    nodeId={selectedNodeId}
                    translationMap={translationMap}
                  />
                  <QuestionPanel
                    record={selectedRecord}
                    annotations={annotations}
                    imageId={selectedImageId}
                    mode="node"
                    nodeId={selectedNodeId}
                    onAnswerChange={handleAnswerChange}
                    translationMap={translationMap}
                  />
                </>
              ) : (
                <QuestionPanel
                  record={selectedRecord}
                  annotations={annotations}
                  imageId={selectedImageId}
                  mode="tree"
                  nodeId={null}
                  onAnswerChange={handleAnswerChange}
                  translationMap={translationMap}
                />
              )}

              {selectedMissingReport &&
              (selectedMissingReport.missing_nodes.length || selectedMissingReport.tree_missing.length) ? (
                <section className="sectionCard">
                  <h2 className="sectionTitle">남은 항목</h2>
                  <div className="pillCloud">
                    {selectedMissingReport.missing_nodes.map((item) => (
                      <span className="finalizePill" key={item.node_id}>{humanLabel(item.node_id)}</span>
                    ))}
                    {selectedMissingReport.tree_missing.length ? (
                      <span className="finalizePill">전체 트리</span>
                    ) : null}
                  </div>
                </section>
              ) : null}
            </>
          ) : null}
        </section>
      </div>
    </main>
  );
}
