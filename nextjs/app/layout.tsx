// @ts-nocheck

import './globals.css';

export const metadata = {
  title: process.env.APP_TITLE || 'H-SAM Review Tool',
  description: 'Next.js reviewer UI for hierarchical segmentation review.',
};

export default function RootLayout({ children }) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
