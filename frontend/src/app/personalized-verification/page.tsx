import { redirect } from 'next/navigation';

export default async function Page() {
  let res;
  try {
    res = await fetch(process.env.NEXT_PUBLIC_API + 'train/status', {
      method: 'GET',
      cache: 'no-store',
    });
  } catch (err) {
    console.log('Error fetching train status:', err);
  }
  if (res?.ok) {
    redirect('/personalized-verification/verify');
  }
  redirect('/personalized-verification/train');
}
