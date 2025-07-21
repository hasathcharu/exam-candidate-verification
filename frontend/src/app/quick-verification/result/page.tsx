'use client';
import Footer from '@/components/footer';
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import NvResults from '@/components/nv-results';
import Link from 'next/link';

export default function App() {
  const router = useRouter();
  const [resetButtonLoading, setResetButtonLoading] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const CONFIDENCE_THRESHOLD = parseFloat(
    process.env.NEXT_PUBLIC_CONFIDENCE_THRESHOLD || '0.7'
  );

  const [data, setData] = useState({
    sigGenuine: false,
    sigConfidence: 0.0,
    writerSame: false,
    writerConfidence: 0.0,
  });
  console.log(data);

  useEffect(() => {
    const result = localStorage.getItem('quick_result');
    if (result) {
      const parsedResult = JSON.parse(result);
      setData(parsedResult);
    } else {
      router.push('/quick-verification');
      return;
    }
    setLoaded(true);
  }, []);

  async function handleReset() {
    setResetButtonLoading(true);
    localStorage.removeItem('quick_result');
    router.push('/quick-verification');
    setResetButtonLoading(false);
  }

  function Recommendation() {
    let title = 'Strong Agreement Between Models”';
    let desc = `
        Both the signature and handwriting verification results agree on the results with high confidence. But if you still have doubts, you can always test with the personalized model for better accuracy.
    `;
    if (
      data.sigConfidence < CONFIDENCE_THRESHOLD ||
      data.writerConfidence < CONFIDENCE_THRESHOLD
    ) {
      desc = `
        The models have low confidence in the results. We recommend testing with the personalized model for better accuracy.
        `;
    }

    if (data.sigGenuine != data.writerSame) {
      desc = `The two models are disagreeing on the results. We recommend testing with the personalized model for better accuracy.`;
    }

    if (
      data.sigConfidence < CONFIDENCE_THRESHOLD ||
      data.writerConfidence < CONFIDENCE_THRESHOLD ||
      data.sigGenuine != data.writerSame
    ) {
      title = 'Test with the Personalized Model';
      return (
        <div className='card bg-[var(--color-warning-content)]   w-xl'>
          <div className='card-body'>
            <h2 className='card-title'>{title}</h2>
            <p className='mb-2 text-md'>{desc}</p>
            <div className='card-actions justify-end'>
              <Link href='/personalized-verification'>
                <button className='btn btn-error'>
                  Train Personalized Model
                </button>
              </Link>
            </div>
          </div>
        </div>
      );
    }
    return (
      <div className='card bg-[var(--color-info-content)]   w-xl'>
        <div className='card-body'>
          <h2 className='card-title'>{title}</h2>
          <p className='mb-2 text-md'>{desc}</p>
          <div className='card-actions justify-end'>
            <Link href='/personalized-verification'>
              <button className='btn btn-accent'>
                Train Personalized Model
              </button>
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    loaded && (
      <div className='flex flex-col min-h-screen pt-10'>
        <main className='flex-grow items-center content-center'>
          <div className='hero h-full'>
            <div className='hero-content'>
              <div className='max-w-[1024px]'>
                <h1 className='text-3xl font-bold text-center mt-10 mb-5'>
                  Quick Writer Verification
                </h1>
                <h2 className='text-xl font-semibold text-center mb-5'>
                  Results Summary
                </h2>
                <NvResults data={data} />
                <br />
                <Recommendation />
                <br />
                <br />
                <div className='text-center w-full justify-center'>
                  {!resetButtonLoading ? (
                    <button
                      className='btn btn-primary btn-soft btn-lg btn-wide'
                      onClick={handleReset}
                    >
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
