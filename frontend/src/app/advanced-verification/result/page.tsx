'use client';
import Footer from '@/components/footer';
import { useState, useEffect } from 'react';
import { toast } from 'sonner';
import { useRouter } from 'next/navigation';
import Link from 'next/link';

export default function App() {
  const router = useRouter();
  const [differentWriter, setDifferentWriter] = useState(false);
  const [resetButtonLoading, setResetButtonLoading] = useState(false);
  const [loaded, setLoaded] = useState(false);
  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch(
          process.env.NEXT_PUBLIC_API + 'results/result.txt',
          {
            method: 'GET',
          }
        );
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        const text = await res.text();
        const line = text.split(/\r?\n/).find((l) => l.trim().length > 0) ?? '';
        if (line === '1') {
          setDifferentWriter(true);
        }
        setLoaded(true);
      } catch (err: any) {
        router.push('/advanced-verification');
        console.log(err);
      }
    };
    fetchData();
  });

  async function handleReset() {
    setResetButtonLoading(true);
    try {
      const res = await fetch(process.env.NEXT_PUBLIC_API + 'reset', {
        method: 'GET',
      });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      router.push('/advanced-verification');
      setResetButtonLoading(false);
    } catch (err: any) {
      toast.error('Resetting model failed. Something went wrong.');
      console.log(err);
      setResetButtonLoading(false);
    }
  }
  return (
    loaded && (
      <div className='flex flex-col h-screen pt-10'>
        <main className='flex-grow'>
          <div className='hero h-full'>
            <div className='hero-content'>
              <div className='max-w-[1024px]'>
                <h1 className='text-3xl font-bold text-center'>
                  Advanced Writer Verification Report
                </h1>
                <br />
                {differentWriter ? (
                  <h2 className='text-2xl font-bold text-center'>
                    Written by{' '}
                    <span className='bg-red-400 py-1 px-2 rounded-sm'>
                      Different Writers
                    </span>
                  </h2>
                ) : (
                  <h2 className='text-2xl font-bold text-center'>
                    Written by the{' '}
                    <span className='bg-green-400 py-1 px-2 rounded-sm'>
                      Same Writer
                    </span>
                  </h2>
                )}
                {differentWriter ? (
                  <p className='py-6 text-md max-w-lg mx-auto'>
                    The sample you provided could not be verified. It appears to
                    be written by someone else. Please review the report below
                    for more details.
                  </p>
                ) : (
                  <p className='py-6 text-md max-w-lg mx-auto'>
                    The sample you provided has been verified to be written by
                    the same writer. You can proceed with further analysis or
                    actions based on the below report.
                  </p>
                )}
                <h2 className='py-6 text-lg font-bold'>Samples</h2>
                <div className='flex gap-10 mb-10'>
                  <div>
                    <h3 className='text-center mb-2 font-bold'>Known Sample</h3>
                    <img
                      src={process.env.NEXT_PUBLIC_API + 'results/known.png'}
                      alt='Known Sample'
                      className='h-full rounded-lg shadow-md mr-2'
                    />
                  </div>
                  <div>
                    <h3 className='text-center mb-2 font-bold'>Test Sample</h3>
                    <img
                      src={process.env.NEXT_PUBLIC_API + 'results/test.png'}
                      alt='Test Sample'
                      className='h-full rounded-lg shadow-md ml-2'
                    />
                  </div>
                </div>

                <h2 className='py-6 text-lg font-bold'>
                  Reconstruction Errors
                </h2>
                <img
                  src={
                    process.env.NEXT_PUBLIC_API +
                    'results/reconstructed_error.png'
                  }
                  alt='Reconstruction Error'
                  className='w-full h-auto rounded-lg shadow-md'
                />
                <h2 className='py-6 text-lg font-bold'>SHAP Waterfall Plot</h2>
                <img
                  src={process.env.NEXT_PUBLIC_API + 'results/waterfall.png'}
                  alt='SHAP Waterfall Plot'
                  className='w-full h-auto rounded-lg shadow-md'
                />
                <h2 className='py-6 text-lg font-bold'>
                  Feature Level Heatmap
                </h2>
                <img
                  src={process.env.NEXT_PUBLIC_API + 'results/heatmap.png'}
                  alt='Reconstruction Error'
                  className='w-full h-auto rounded-lg shadow-md'
                />
                <br />
                <br />
                <div className='text-center w-full justify-center'>
                  <Link href='/advanced-verification'>
                    <button className='btn btn-primary btn-soft btn-lg btn-wide'>
                      <svg
                        xmlns='http://www.w3.org/2000/svg'
                        width='24px'
                        height='24px'
                        viewBox='0 0 24 24'
                        fill='none'
                      >
                        <path
                          d='M6.8008 11.7834L8.07502 11.9256C9.09772 12.0398 9.90506 12.8507 10.0187 13.8779C10.1062 14.6689 10.6104 15.3515 11.3387 15.665L13 16.3547M13 16.3547L9.48838 19.8818C8.00407 21.3727 5.59754 21.3727 4.11323 19.8818C2.62892 18.391 2.62892 15.9738 4.11323 14.4829L14.8635 3.68504L20.2387 9.08398L18.429 10.9017M13 16.3547L16 13.3414M21 9.84867L14.1815 3'
                          stroke='currentColor'
                          strokeWidth='2'
                          strokeLinecap='round'
                        />
                      </svg>
                      Test Another Sample
                    </button>
                  </Link>
                  <br />
                  <br />
                  {!resetButtonLoading ? (
                    <button
                      className='btn btn-ghost btn-soft btn-lg btn-wide'
                      onClick={handleReset}
                    >
                      <svg
                        width='24'
                        height='24'
                        viewBox='0 0 15 15'
                        fill='none'
                        xmlns='http://www.w3.org/2000/svg'
                      >
                        <path
                          d='M5.5 1C5.22386 1 5 1.22386 5 1.5C5 1.77614 5.22386 2 5.5 2H9.5C9.77614 2 10 1.77614 10 1.5C10 1.22386 9.77614 1 9.5 1H5.5ZM3 3.5C3 3.22386 3.22386 3 3.5 3H5H10H11.5C11.7761 3 12 3.22386 12 3.5C12 3.77614 11.7761 4 11.5 4H11V12C11 12.5523 10.5523 13 10 13H5C4.44772 13 4 12.5523 4 12V4L3.5 4C3.22386 4 3 3.77614 3 3.5ZM5 4H10V12H5V4Z'
                          fill='currentColor'
                          fillRule='evenodd'
                          clipRule='evenodd'
                        ></path>
                      </svg>
                      Retrain Model
                    </button>
                  ) : (
                    <button
                      className='btn btn-ghost btn-soft btn-lg btn-wide'
                      onClick={handleReset}
                      disabled
                    >
                      <span className='loading loading-infinity'></span>
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
        </main>
        <Footer />
      </div>
    )
  );
}
