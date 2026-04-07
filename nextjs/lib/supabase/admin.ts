// @ts-nocheck

import { createClient } from '@supabase/supabase-js';

let adminClient: any = null;

export function getSupabaseAdmin() {
  if (adminClient) return adminClient;

  const url = process.env.SUPABASE_URL?.trim();
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY?.trim();

  if (!url) throw new Error('SUPABASE_URL is required.');
  if (!key) throw new Error('SUPABASE_SERVICE_ROLE_KEY is required.');

  adminClient = createClient(url, key, {
    auth: {
      persistSession: false,
      autoRefreshToken: false,
    },
  });

  return adminClient;
}

export function getBucketName() {
  return process.env.SUPABASE_BUCKET?.trim() || 'review-dataset';
}

export function getReviewTableName() {
  return process.env.SUPABASE_REVIEW_TABLE?.trim() || 'review_annotations';
}

export function getAssignmentsTableName() {
  return process.env.SUPABASE_ASSIGNMENTS_TABLE?.trim() || 'review_assignments';
}

export function getManifestTableName() {
  return process.env.SUPABASE_MANIFEST_TABLE?.trim() || 'image_manifest';
}
