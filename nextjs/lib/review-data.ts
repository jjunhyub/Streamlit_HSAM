// @ts-nocheck

import { getAssignmentsTableName, getBucketName, getManifestTableName, getReviewTableName, getSupabaseAdmin } from './supabase/admin';
import { nowIso, safeToken } from './utils';

const assetAllowCache = new Map();
const ASSET_ALLOW_TTL_MS = 1000 * 60 * 5;

export function annotationKeyForReviewer(reviewerId: string) {
  return safeToken((reviewerId || '').trim() || 'anonymous');
}

export async function loadAnnotationsForReviewer(reviewerId: string) {
  const key = annotationKeyForReviewer(reviewerId);
  const supabase = getSupabaseAdmin();
  const { data, error } = await supabase
    .from(getReviewTableName())
    .select('payload')
    .eq('reviewer_id', key)
    .limit(1);

  if (error) throw error;
  const payload = data?.[0]?.payload;
  return payload && typeof payload === 'object' ? payload : {};
}

export async function saveAnnotationsForReviewer(reviewerId: string, payload: any) {
  const key = annotationKeyForReviewer(reviewerId);
  const supabase = getSupabaseAdmin();
  const { error } = await supabase
    .from(getReviewTableName())
    .upsert({
      reviewer_id: key,
      payload,
      updated_at: nowIso(),
    });

  if (error) throw error;
}

export async function loadAssignedImageIds(reviewerId: string) {
  const supabase = getSupabaseAdmin();
  const { data, error } = await supabase
    .from(getAssignmentsTableName())
    .select('image_id, sort_index')
    .eq('reviewer_id', reviewerId)
    .order('sort_index');

  if (error) throw error;
  return (data || []).map((row) => ({ image_id: row.image_id, sort_index: row.sort_index }));
}

export async function loadRecordsForReviewer(reviewerId: string) {
  const assigned = await loadAssignedImageIds(reviewerId);
  const imageIds = assigned.map((row) => row.image_id);
  if (!imageIds.length) return {};

  const supabase = getSupabaseAdmin();
  const { data, error } = await supabase
    .from(getManifestTableName())
    .select('*')
    .in('image_id', imageIds);

  if (error) throw error;

  const byId = Object.fromEntries((data || []).map((row) => [row.image_id, row]));
  const ordered = {};
  for (const imageId of imageIds) {
    if (byId[imageId]) ordered[imageId] = byId[imageId];
  }
  return ordered;
}

export async function loadReviewPageData(reviewerId: string) {
  const [records, annotations] = await Promise.all([
    loadRecordsForReviewer(reviewerId),
    loadAnnotationsForReviewer(reviewerId),
  ]);

  return { records, annotations };
}

export function collectAllowedAssetPaths(record: any) {
  const paths = new Set<string>();

  const add = (value: any) => {
    if (value) paths.add(String(value));
  };

  const deriveSiblingFromMask = (maskPath: any, nextSuffix: string) => {
    if (!maskPath || typeof maskPath !== 'string') return null;
    if (!/\.mask\.png$/i.test(maskPath)) return null;
    return maskPath.replace(/\.mask\.png$/i, nextSuffix);
  };

  add(record?.root_image_path);
  add(record?.root_overlay_path);

  for (const nodeId of record?.actual_nodes || []) {
    const node = record?.nodes?.[nodeId] || {};
    const maskPath = node.mask_path || null;

    add(maskPath);
    add(node.overlay_path);
    add(node.mask_original_path);
    add(node.mask_original_full_path);
    add(node.instances_colored_path);

    // // DB에 필드가 없더라도 mask_path 기준으로 유도해서 허용
    // add(deriveSiblingFromMask(maskPath, '.overlay.png'));
    // add(deriveSiblingFromMask(maskPath, '.mask.original.full.png'));
    // add(deriveSiblingFromMask(maskPath, '.instances.colored.png'));

    for (const instancePath of node.instance_paths || []) {
      add(instancePath);
    }
  }

  return paths;
}

export async function reviewerCanAccessImage(reviewerId: string, imageId: string) {
  const supabase = getSupabaseAdmin();
  const { data, error } = await supabase
    .from(getAssignmentsTableName())
    .select('image_id')
    .eq('reviewer_id', reviewerId)
    .eq('image_id', imageId)
    .limit(1);

  if (error) throw error;
  return Boolean(data?.length);
}

export async function loadSingleRecord(imageId: string) {
  const supabase = getSupabaseAdmin();
  const { data, error } = await supabase
    .from(getManifestTableName())
    .select('*')
    .eq('image_id', imageId)
    .limit(1)
    .maybeSingle();

  if (error) throw error;
  return data || null;
}

export async function getAllowedPathsForReviewerImage(reviewerId: string, imageId: string) {
  const cacheKey = `${reviewerId}::${imageId}`;
  const cached = assetAllowCache.get(cacheKey);
  if (cached && cached.expiresAt > Date.now()) {
    return cached.paths;
  }

  const canAccess = await reviewerCanAccessImage(reviewerId, imageId);
  if (!canAccess) return null;

  const record = await loadSingleRecord(imageId);
  if (!record) return null;

  const paths = collectAllowedAssetPaths(record);
  assetAllowCache.set(cacheKey, {
    expiresAt: Date.now() + ASSET_ALLOW_TTL_MS,
    paths,
  });

  return paths;
}

export async function downloadAssetFromStorage(path: string) {
  const supabase = getSupabaseAdmin();
  const { data, error } = await supabase.storage.from(getBucketName()).download(path);
  if (error) throw error;
  return data;
}
