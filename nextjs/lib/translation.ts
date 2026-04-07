// @ts-nocheck

export function parseTranslationPayload(payload) {
  const out = {};
  const images = Array.isArray(payload?.images) ? payload.images : [];

  for (const item of images) {
    if (!item || typeof item !== 'object') continue;
    const imageId = String(item.image_id || '');
    const nodes = item.nodes;
    if (!imageId || !nodes || typeof nodes !== 'object') continue;

    out[imageId] ||= {};
    for (const [nodeId, info] of Object.entries(nodes)) {
      if (!info || typeof info !== 'object') continue;
      out[imageId][String(nodeId)] = {
        en: String(info.en || ''),
        ko: String(info.ko || ''),
      };
    }
  }

  return out;
}
