// @ts-nocheck

export function safeToken(value: string) {
  return String(value || '').replace(/[^a-zA-Z0-9_-]/g, '_');
}

export function humanLabel(nodeId: string) {
  const raw = String(nodeId || '');
  return raw.includes('__') ? raw.split('__').at(-1) || raw : raw;
}

export function nowIso() {
  return new Date().toISOString().replace(/\.\d{3}Z$/, 'Z');
}

export function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

export function chunk(items: any[], size: number) {
  const out = [];
  for (let i = 0; i < items.length; i += size) {
    out.push(items.slice(i, i + size));
  }
  return out;
}

export function base64UrlEncode(input: string | Buffer) {
  const buf = Buffer.isBuffer(input) ? input : Buffer.from(String(input));
  return buf
    .toString('base64')
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/g, '');
}

export function base64UrlDecode(input: string) {
  const normalized = String(input || '')
    .replace(/-/g, '+')
    .replace(/_/g, '/');
  const padding = normalized.length % 4 === 0 ? '' : '='.repeat(4 - (normalized.length % 4));
  return Buffer.from(normalized + padding, 'base64').toString('utf8');
}

export function guessContentType(pathname: string) {
  const path = String(pathname || '').toLowerCase();
  if (path.endsWith('.png')) return 'image/png';
  if (path.endsWith('.jpg') || path.endsWith('.jpeg')) return 'image/jpeg';
  if (path.endsWith('.webp')) return 'image/webp';
  if (path.endsWith('.gif')) return 'image/gif';
  if (path.endsWith('.bmp')) return 'image/bmp';
  if (path.endsWith('.svg')) return 'image/svg+xml';
  if (path.endsWith('.tif') || path.endsWith('.tiff')) return 'image/tiff';
  if (path.endsWith('.json')) return 'application/json';
  return 'application/octet-stream';
}

export function toErrorMessage(error: unknown) {
  if (error instanceof Error) return error.message;
  return String(error || 'Unknown error');
}
