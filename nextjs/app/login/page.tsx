// @ts-nocheck

import { redirect } from 'next/navigation';
import LoginForm from '../../components/login-form';
import { getSession } from '../../lib/auth/session';

export const dynamic = 'force-dynamic';

export default async function LoginPage() {
  const session = await getSession();
  if (session) {
    redirect('/');
  }

  return (
    <main className="loginPage">
      <LoginForm />
    </main>
  );
}
