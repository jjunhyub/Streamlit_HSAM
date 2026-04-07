// @ts-nocheck
'use client';

import { useMemo } from 'react';
import { getInspectorPills, nodeAssets } from '../lib/review-logic';

function buildAssetUrl(imageId, path) {
  if (!path) return '';
  const params = new URLSearchParams({ imageId: String(imageId), path: String(path) });
  return `/api/assets?${params.toString()}`;
}

function FigureCard({ title, children }) {
  return (
    <div className="figureCard">
      <div className="figureTitle">{title}</div>
      {children}
    </div>
  );
}

function AspectFrame({ width, height, children, dark = true }) {
  const w = Math.max(1, Number(width || 1));
  const h = Math.max(1, Number(height || 1));
  return (
    <div className={`aspectFrame ${dark ? 'isDark' : ''}`} style={{ aspectRatio: `${w} / ${h}` }}>
      {children}
    </div>
  );
}

function DirectImageFigure({ imageId, path, title, fullSize }) {
  if (!path) return null;
  const url = buildAssetUrl(imageId, path);
  return (
    <FigureCard title={title}>
      <AspectFrame width={fullSize?.[0]} height={fullSize?.[1]}>
        <img className="frameImage" src={url} alt={title} loading="lazy" />
      </AspectFrame>
    </FigureCard>
  );
}

export default function VisualsPanel({ record, nodeId, translationMap }) {
  const assets = useMemo(() => nodeAssets(record, nodeId), [record, nodeId]);
  console.log('assets', assets);
  const leaf = String(nodeId || '').split('__').at(-1) || nodeId;
  const pills = useMemo(
    () => getInspectorPills(record, nodeId, translationMap),
    [record, nodeId, translationMap]
  );

  const reviewFigures = [
    assets.root_original ? (
      <DirectImageFigure
        key="root-original"
        imageId={record.image_id}
        path={assets.root_original}
        title="원본 이미지"
        fullSize={assets.full_size}
      />
    ) : null,

    assets.overlay ? (
      <DirectImageFigure
        key="overlay"
        imageId={record.image_id}
        path={assets.overlay}
        title={`오버레이 <${leaf}>`}
        fullSize={assets.full_size}
      />
    ) : null,

    assets.mask_original_full ? (
      <DirectImageFigure
        key="mask-original-full"
        imageId={record.image_id}
        path={assets.mask_original_full}
        title={`원본 <${leaf}>`}
        fullSize={assets.full_size}
      />
    ) : null,

    assets.mask ? (
      <DirectImageFigure
        key="mask"
        imageId={record.image_id}
        path={assets.mask}
        title={`마스크 <${leaf}>`}
        fullSize={assets.full_size}
      />
    ) : null,

    assets.instances_colored ? (
      <DirectImageFigure
        key="instances-colored"
        imageId={record.image_id}
        path={assets.instances_colored}
        title={`인스턴스 <${leaf}>`}
        fullSize={assets.full_size}
      />
    ) : null,
  ].filter(Boolean);

  return (
    <section className="sectionCard">
      <div className="sectionHeaderWithMeta">
        <div>
          <h2 className="sectionTitle">Visuals</h2>
          <div className="statusPillsRow">
            {pills.map((pill) => (
              <span className="statusPill" key={pill}>{pill}</span>
            ))}
          </div>
        </div>
      </div>

      {reviewFigures.length ? (
        <div className="visualsGrid">{reviewFigures}</div>
      ) : (
        <div className="emptyBox">노드 검토용 이미지를 찾지 못했습니다.</div>
      )}
    </section>
  );
}