'use client';
import Footer from '@/components/footer';
import { useState, useEffect } from 'react';
import { toast } from 'sonner';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import Markdown from '@/components/ui/markdown';
import ImageViewer from '@/components/ui/image-viewer';
import Emphasis from '@/components/ui/emphasis';
import PvResults from '@/components/pv-results';

export default function App() {
  const router = useRouter();
  const [differentWriter, setDifferentWriter] = useState(false);
  const [description, setDescription] = useState('');
  const [resetButtonLoading, setResetButtonLoading] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [resultsAvailable, setResultsAvailable] = useState(false);
  const [currentResults, setCurrentResults] = useState({
    sigGenuine: false,
    sigConfidence: 0.0,
    writerSame: false,
    writerConfidence: 0.0,
    personalizedWriterSame: false,
    personalizedWriterConfidence: 0.0,
  });

  useEffect(() => {
    let storage = localStorage.getItem('quick_result');
    let result: any = {};
    if (storage) {
      setResultsAvailable(true);
      result = JSON.parse(storage || '{}');
    } else {
      return;
    }
    const fetchData = async () => {
      try {
        const res = await fetch(
          process.env.NEXT_PUBLIC_API + 'results/result.json',
          {
            method: 'GET',
          }
        );
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        const response = await res.json();
        setDescription(response.description || 'No description available.');
        if (response.same_writer === 0) {
          setDifferentWriter(true);
        }
        result['personalizedWriterSame'] = response.same_writer;
        result['personalizedWriterConfidence'] = response.confidence;
        setCurrentResults(result);
        setLoaded(true);
      } catch (err: any) {
        router.push('/personalized-verification');
        console.log(err);
      }
    };
    fetchData();
  }, []);

  async function handleReset() {
    setResetButtonLoading(true);
    try {
      const res = await fetch(process.env.NEXT_PUBLIC_API + 'reset', {
        method: 'GET',
      });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      router.push('/personalized-verification');
      setResetButtonLoading(false);
    } catch (err: any) {
      toast.error('Resetting model failed. Something went wrong.');
      console.log(err);
      setResetButtonLoading(false);
    }
  }
  return (
    loaded && (
      <div className='flex flex-col min-h-screen pt-10'>
        <main className='flex-grow'>
          <div className='hero h-full'>
            <div className='hero-content'>
              <div className='max-w-[1024px]'>
                <h1 className='text-3xl font-bold text-center mt-10'>
                  Personalized Writer Verification Report
                </h1>
                <br />
                {differentWriter ? (
                  <h2 className='text-2xl font-bold text-center'>
                    Written by <Emphasis type='red'>Different Writers</Emphasis>
                  </h2>
                ) : (
                  <h2 className='text-2xl font-bold text-center'>
                    Written by the <Emphasis type='green'>Same Writer</Emphasis>
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
                <div className='flex items-center justify-center'>
                  {resultsAvailable && <PvResults data={currentResults} />}
                </div>
                <h2 className='py-6 text-lg font-bold max-w-3xl mx-auto'>
                  Samples
                </h2>
                <div className='flex justify-between max-w-3xl px-4 font-bold mb-2'>
                  <span>Known Sample</span>
                  <span>Test Sample</span>
                </div>
                <div className='flex gap-10 mb-3 max-w-3xl mx-auto'>
                  <figure className='diff aspect-16/9 rounded-3xl' tabIndex={0}>
                    <div className='diff-item-1' role='img' tabIndex={0}>
                      <img
                        alt='Known Sample'
                        src={process.env.NEXT_PUBLIC_API + 'results/known.png'}
                      />
                    </div>
                    <div className='diff-item-2' role='img'>
                      <img
                        alt='Test Sample'
                        src={process.env.NEXT_PUBLIC_API + 'results/test.png'}
                      />
                    </div>
                    <div className='diff-resizer'></div>
                  </figure>
                </div>
                <h2 className='py-6 pb-1 text-lg mx-auto font-bold max-w-3xl'>
                  <Emphasis type='ai'>Personalized Explanation</Emphasis>
                </h2>
                <div className='py-6 mx-auto'>
                  <Markdown text={description} />
                </div>
                <h2 className='py-6 text-lg font-bold max-w-3xl mx-auto'>
                  Reconstruction Errors
                </h2>
                <img
                  src={
                    process.env.NEXT_PUBLIC_API +
                    'results/reconstructed_error.png'
                  }
                  alt='Reconstruction Error'
                  className='w-full h-auto rounded-lg shadow-md max-w-3xl mx-auto'
                />
                <h2 className='py-6 text-lg font-bold  max-w-3xl mx-auto'>
                  Most Anomalous Features
                </h2>
                <img
                  src={process.env.NEXT_PUBLIC_API + 'results/pos_waterfall.png'}
                  alt='Most Anomalous Features'
                  className='w-full h-auto rounded-lg shadow-md  max-w-3xl mx-auto'
                />
                <h2 className='py-6 text-lg font-bold  max-w-3xl mx-auto'>
                  Most Normal Features
                </h2>
                <img
                  src={process.env.NEXT_PUBLIC_API + 'results/neg_waterfall.png'}
                  alt='Most Normal Features'
                  className='w-full h-auto rounded-lg shadow-md  max-w-3xl mx-auto'
                />
                <h2 className='py-6 text-lg font-bold  max-w-3xl mx-auto'>
                  Anomalous Feature Heatmap
                </h2>
                <img
                  src={process.env.NEXT_PUBLIC_API + 'results/pos_heatmap.png'}
                  alt='Anomalous Feature Heatmap'
                  className='w-full h-auto rounded-lg shadow-md  max-w-3xl mx-auto'
                />
                <h2 className='py-6 text-lg font-bold  max-w-3xl mx-auto'>
                  Normal Feature Heatmap
                </h2>
                <img
                  src={process.env.NEXT_PUBLIC_API + 'results/ng_heatmap.png'}
                  alt='Normal Feature Heatmap'
                  className='w-full h-auto rounded-lg shadow-md  max-w-3xl mx-auto'
                />
                <br />
                <br />
                <div className='text-center w-full justify-center'>
                  <Link href='/personalized-verification'>
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
