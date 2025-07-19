// app/auto-start/page.tsx
import { redirect } from 'next/navigation';

export default async function Page() {
  try {
    const res = await fetch(process.env.NEXT_PUBLIC_API + 'train/status', {
      method: 'GET',
      cache: 'no-store',
    });
    if (res.ok) {
      return redirect('/personalized-verification/verify');
    }
  } catch (err) {
    console.error('Error fetching train status:', err);
  }
  redirect('/personalized-verification/train');
}
