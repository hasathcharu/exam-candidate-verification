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
import { FlaskConical, Trash2 } from 'lucide-react';
import QuickAlert from '@/components/quick-alert';

export default function App() {
  const router = useRouter();
  const [differentWriter, setDifferentWriter] = useState(false);
  const [personalizedWriterSame, setPersonalizedWriterSame] = useState(false);
  const [description, setDescription] = useState('');
  const [resetButtonLoading, setResetButtonLoading] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [resultsAvailable, setResultsAvailable] = useState(false);
  const [qVOpen, setQVOpen] = useState(false);
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
      router.push('/quick-verification');
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
        } else {
          setPersonalizedWriterSame(true);
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
                {personalizedWriterSame ? (
                  <>
                    <h2 className='py-6 text-lg font-bold  max-w-3xl mx-auto'>
                      Most Normal Features
                    </h2>
                    <img
                      src={
                        process.env.NEXT_PUBLIC_API +
                        'results/neg_waterfall.png'
                      }
                      alt='Most Normal Features'
                      className='w-full h-auto rounded-lg shadow-md  max-w-3xl mx-auto'
                    />
                    <h2 className='py-6 text-lg font-bold  max-w-3xl mx-auto'>
                      Normal Feature Heatmap
                    </h2>
                    <img
                      src={
                        process.env.NEXT_PUBLIC_API + 'results/neg_heatmap.png'
                      }
                      alt='Normal Feature Heatmap'
                      className='w-full h-auto rounded-lg shadow-md  max-w-3xl mx-auto'
                    />
                  </>
                ) : (
                  <>
                    <h2 className='py-6 text-lg font-bold  max-w-3xl mx-auto'>
                      Most Anomalous Features
                    </h2>
                    <img
                      src={
                        process.env.NEXT_PUBLIC_API +
                        'results/pos_waterfall.png'
                      }
                      alt='Most Anomalous Features'
                      className='w-full h-auto rounded-lg shadow-md  max-w-3xl mx-auto'
                    />
                    <h2 className='py-6 text-lg font-bold  max-w-3xl mx-auto'>
                      Anomalous Feature Heatmap
                    </h2>
                    <img
                      src={
                        process.env.NEXT_PUBLIC_API + 'results/pos_heatmap.png'
                      }
                      alt='Anomalous Feature Heatmap'
                      className='w-full h-auto rounded-lg shadow-md  max-w-3xl mx-auto'
                    />
                  </>
                )}
                <br />
                <br />
                <div className='text-center w-full justify-center'>
                  <Link href='/personalized-verification'>
                    <button className='btn btn-primary btn-soft btn-lg btn-wide'>
                      <FlaskConical /> Test Another Sample
                    </button>
                  </Link>
                  <br />
                  <br />
                  {!resetButtonLoading ? (
                    <button
                      className='btn btn-info btn-soft btn-lg btn-wide'
                      onClick={handleReset}
                    >
                      <Trash2 /> Retrain Model
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
