// @ts-nocheck
'use client';

import { useMemo } from 'react';
import {
  TREE_BUTTON_HEIGHT_PX,
  TREE_ROW_GAP_PX,
  TREE_ROW_HEIGHT_PX,
  TREE_SUMMARY_NODE_ID,
  TREE_PANEL_TOP_PAD_PX,
  TREE_BUTTON_MAX_WIDTH_PX,
  TREE_BUTTON_MIN_WIDTH_PX,
  TREE_BUTTON_BASE_PX,
  TREE_CHAR_WIDTH_PX,
  allNodesConfirmed,
  buildTreeConnectorPaths,
  computeHierarchyLayout,
  injectTreeSummaryNode,
  isReviewableNode,
  nodeConfirmed,
  nodeProgress,
  translatedLabel,
  treeSummaryConfirmed,
} from '../lib/review-logic';

function buttonWidthForLabel(label) {
  const text = String(label || '');
  const width = TREE_BUTTON_BASE_PX + TREE_CHAR_WIDTH_PX * text.length + 12;
  return Math.max(TREE_BUTTON_MIN_WIDTH_PX, Math.min(TREE_BUTTON_MAX_WIDTH_PX, width));
}

function getNodeLabel(imageId, nodeId, translationMap) {
  return nodeId === TREE_SUMMARY_NODE_ID ? 'Full-Tree' : translatedLabel(imageId, nodeId, translationMap);
}

export default function HierarchyTree({
  record,
  annotations,
  selectedMode,
  selectedNodeId,
  onSelectNode,
  onSelectTreeSummary,
  translationMap,
}) {
  const imageId = record?.image_id;
  const [done, total] = nodeProgress(annotations, imageId, record);
  const treeDone = treeSummaryConfirmed(annotations, imageId);
  const treeEnabled = allNodesConfirmed(annotations, imageId, record);

  const layoutRecord = useMemo(() => injectTreeSummaryNode(record), [record]);

  const layout = useMemo(
    () =>
      computeHierarchyLayout(layoutRecord, {
        imageId,
        translationMap,
        getNodeLabel,
        getNodeWidth: (nodeId) =>
          buttonWidthForLabel(getNodeLabel(imageId, nodeId, translationMap)),
      }),
    [layoutRecord, imageId, translationMap]
  );

  const connectorPaths = useMemo(
    () => buildTreeConnectorPaths(layoutRecord, layout),
    [layoutRecord, layout]
  );

  const svgHeight =
    TREE_PANEL_TOP_PAD_PX * 2 +
    layout.rows.length * TREE_ROW_HEIGHT_PX +
    Math.max(0, layout.rows.length - 1) * TREE_ROW_GAP_PX;

  return (
    <section
      className="sectionCard hierarchyCard"
      style={{ minWidth: 0, maxWidth: '100%', overflow: 'hidden' }}
    >
      <div className="sectionHeaderWithMeta">
        <div>
          <h2 className="sectionTitle">Hierarchy View</h2>
          <div className="sectionSubtle">
            <strong>{imageId}</strong> · 노드 완료 {done}/{total} · 전체 트리 질문 {treeDone ? '완료' : '미완료'}
          </div>
        </div>
      </div>

      <div
        className="treeScroller"
        style={{
          width: '100%',
          maxWidth: '100%',
          minWidth: 0,
          overflowX: 'auto',
          overflowY: 'hidden',
        }}
      >
        <div
          className="treeCanvas"
          style={{
            width: layout.treeWidth,
            minWidth: '100%',
            height: svgHeight,
            position: 'relative',
            flex: '0 0 auto',
          }}
        >
          <svg
            className="treeSvg"
            width={layout.treeWidth}
            height={svgHeight}
            viewBox={`0 0 ${layout.treeWidth} ${svgHeight}`}
            aria-hidden="true"
          >
            <g className="treeConnectorGroup">
              {connectorPaths.map((path, index) => (
                <path key={index} d={path} />
              ))}
            </g>
          </svg>

          {layout.rows.flat().map((nodeId) => {
            const rowIndex = layout.rows.findIndex((row) => row.includes(nodeId));
            const top =
              TREE_PANEL_TOP_PAD_PX +
              rowIndex * (TREE_ROW_HEIGHT_PX + TREE_ROW_GAP_PX) +
              (TREE_ROW_HEIGHT_PX - TREE_BUTTON_HEIGHT_PX) / 2;

            const left = layout.nodeLefts[nodeId];
            const label = getNodeLabel(imageId, nodeId, translationMap);
            const width = Math.max(layout.nodeWidths[nodeId] || 0, buttonWidthForLabel(label));


            if (nodeId === TREE_SUMMARY_NODE_ID) {
              return (
                <button
                  key={nodeId}
                  type="button"
                  className={[
                    'treeNodeButton',
                    selectedMode === 'tree' ? 'isSelected' : '',
                    treeDone ? 'isDone' : '',
                    !treeEnabled ? 'isMuted' : '',
                  ].join(' ')}
                  style={{ left, top, width }}
                  disabled={!treeEnabled}
                  onClick={onSelectTreeSummary}
                >
                  {label}
                </button>
              );
            }

            const actualNode = record.nodes?.[nodeId];
            const isActual = Boolean(actualNode?.actual);
            const isClickable = isActual && isReviewableNode(nodeId);
            const isSelected = selectedMode === 'node' && selectedNodeId === nodeId;
            const isDone = isActual ? nodeConfirmed(annotations, record, imageId, nodeId) : false;

            return (
              <button
                key={nodeId}
                type="button"
                className={[
                  'treeNodeButton',
                  isSelected ? 'isSelected' : '',
                  isDone ? 'isDone' : '',
                  !isClickable ? 'isMuted' : '',
                ].join(' ')}
                style={{ left, top, width }}
                disabled={!isClickable}
                onClick={() => onSelectNode(nodeId)}
                title={label}
              >
                {label}
              </button>
            );
          })}
        </div>
      </div>

      {/* {!treeEnabled ? (
        <div className="sectionHint">모든 노드를 완료해야 전체 트리 질문이 활성화됩니다.</div>
      ) : null} */}
    </section>
  );
}