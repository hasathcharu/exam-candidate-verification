'use client';
import Footer from '@/components/footer';
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import Markdown from '@/components/ui/markdown';
import Emphasis from '@/components/ui/emphasis';
import PvResults from '@/components/pv-results';
import { FlaskConical, Sparkles, Trash2 } from 'lucide-react';
import { useParams } from 'next/navigation';

export default function App() {
  const router = useRouter();
  const params = useParams();
  const taskId = params.taskId;
  const [sameWriter, setSameWriter] = useState(false);
  const [description, setDescription] = useState('');
  const [loaded, setLoaded] = useState(false);
  const [currentResults, setCurrentResults] = useState({
    sigGenuine: false,
    sigConfidence: 0.0,
    writerSame: false,
    writerConfidence: 0.0,
    personalizedWriterSame: false,
    personalizedWriterConfidence: 0.0,
  });

  useEffect(() => {
    const fetchData = async () => {
      try {
        const quickRes = await fetch(
          process.env.NEXT_PUBLIC_API + `results/${taskId}/quick_result.json`,
          {
            method: 'GET',
          }
        );
        if (!quickRes.ok)
          throw new Error(`${quickRes.status} ${quickRes.statusText}`);
        const result = await quickRes.json();
        const res = await fetch(
          process.env.NEXT_PUBLIC_API + `results/${taskId}/result.json`,
          {
            method: 'GET',
          }
        );
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        const response = await res.json();
        setDescription(response.description || 'No description available.');
        if (response.same_writer === true) {
          result['personalizedWriterSame'] = true;
        }
        const atLeastTwoTrue =
          [result.sigGenuine, result.writerSame, result.sameWriter].filter(
            Boolean
          ).length >= 2;
        console.log(atLeastTwoTrue);

        if (atLeastTwoTrue) {
          setSameWriter(true);
        }
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
                {sameWriter ? (
                  <>
                    <h2 className='text-2xl font-bold text-center'>
                      Written by the{' '}
                      <Emphasis type='green'>Same Writer</Emphasis>
                    </h2>
                    <p className='py-6 text-md max-w-lg mx-auto'>
                      The sample you provided has been verified to be written by
                      the same writer. You can proceed with further analysis or
                      actions based on the below report.
                    </p>
                  </>
                ) : (
                  <>
                    <h2 className='text-2xl font-bold text-center'>
                      Written by{' '}
                      <Emphasis type='red'>Different Writers</Emphasis>
                    </h2>
                    <p className='py-6 text-md max-w-lg mx-auto'>
                      The sample you provided could not be verified. It appears
                      to be written by someone else. Please review the report
                      below for more details.
                    </p>
                  </>
                )}
                <div className='flex items-center justify-center'>
                  <PvResults data={currentResults} />
                </div>
                <h2 className='py-6 text-lg font-bold max-w-3xl mx-auto'>
                  Signature Sample
                </h2>
                <img
                  alt='Signature Sample'
                  src={
                    process.env.NEXT_PUBLIC_API +
                    `results/${taskId}/test-sig.png`
                  }
                  className='w-72 h-32 rounded-3xl object-fill'
                />
                <h2 className='py-6 text-lg font-bold max-w-3xl mx-auto'>
                  Handwriting Samples
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
                        src={
                          process.env.NEXT_PUBLIC_API +
                          `results/${taskId}/known.png`
                        }
                      />
                    </div>
                    <div className='diff-item-2' role='img'>
                      <img
                        alt='Test Sample'
                        src={
                          process.env.NEXT_PUBLIC_API +
                          `results/${taskId}/test.png`
                        }
                      />
                    </div>
                    <div className='diff-resizer'></div>
                  </figure>
                </div>
                <h2 className='py-6 pb-1 text-lg mx-auto font-bold max-w-3xl'>
                  <Emphasis type='ai'>
                    <Sparkles size={20} /> Personalized Explanation
                  </Emphasis>
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
                    `results/${taskId}/reconstructed_error.png`
                  }
                  alt='Reconstruction Error'
                  className='w-full h-auto rounded-lg shadow-md max-w-3xl mx-auto'
                />
                {currentResults.personalizedWriterSame ? (
                  <>
                    <h2 className='py-6 text-lg font-bold  max-w-3xl mx-auto'>
                      Most Normal Features
                    </h2>
                    <img
                      src={
                        process.env.NEXT_PUBLIC_API +
                        `results/${taskId}/neg_waterfall.png`
                      }
                      alt='Most Normal Features'
                      className='w-full h-auto rounded-lg shadow-md  max-w-3xl mx-auto'
                    />
                    <h2 className='py-6 text-lg font-bold  max-w-3xl mx-auto'>
                      Normal Feature Heatmap
                    </h2>
                    <img
                      src={
                        process.env.NEXT_PUBLIC_API +
                        `results/${taskId}/neg_heatmap.png`
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
                        `results/${taskId}/pos_waterfall.png`
                      }
                      alt='Most Anomalous Features'
                      className='w-full h-auto rounded-lg shadow-md  max-w-3xl mx-auto'
                    />
                    <h2 className='py-6 text-lg font-bold  max-w-3xl mx-auto'>
                      Anomalous Feature Heatmap
                    </h2>
                    <img
                      src={
                        process.env.NEXT_PUBLIC_API +
                        `results/${taskId}/pos_heatmap.png`
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
