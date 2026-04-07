// @ts-nocheck
'use client';

import { imageComplete, nodeProgress } from '../lib/review-logic';

export default function ImageList({
  records,
  annotations,
  selectedImageId,
  imageSearch,
  onImageSearch,
  onSelectImage,
}) {
  const imageIds = Object.keys(records || {});
  const filteredIds = imageIds.filter((imageId) => imageId.toLowerCase().includes(String(imageSearch || '').toLowerCase()));

  return (
    <aside className="leftSidebar">
      <div className="sidebarSectionHeader">Images</div>
      <input
        className="textField sidebarSearch"
        value={imageSearch}
        onChange={(event) => onImageSearch(event.target.value)}
        placeholder="image id 검색"
      />
      <div className="sidebarCount">{filteredIds.length}개 표시</div>

      <div className="imageListScroller">
        {filteredIds.map((imageId) => {
          const record = records[imageId];
          const [done, total] = nodeProgress(annotations, imageId, record);
          const completed = imageComplete(annotations, imageId, record);
          const selected = selectedImageId === imageId;
          const progress = total > 0 ? (done / total) * 100 : 0;
          const displayId = /^\d+$/.test(String(imageId)) ? String(Number(imageId)) : String(imageId);
          const icon = completed ? '🟦' : done > 0 ? '🟡' : '⬜';

          return (
            <button
              key={imageId}
              type="button"
              className={[
                'imageListItem',
                selected ? 'isSelected' : '',
                completed ? 'isDone' : '',
              ].join(' ')}
              onClick={() => onSelectImage(imageId)}
            >
              <div className="imageListItemTitle">{icon} {displayId}</div>
              <div className="imageListItemMeta">노드 진행률: {done}/{total}</div>
              <div className="imageListItemMeta">{progress.toFixed(1)}% 완료</div>
            </button>
          );
        })}
      </div>
    </aside>
  );
}
