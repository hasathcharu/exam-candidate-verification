'use client';
import Footer from '@/components/footer';
import { FileUploadComponent } from '@/components/file-upload-component';
import { useEffect, useState } from 'react';
import { toast } from 'sonner';
import { useRouter } from 'next/navigation';
import LoadingDialog from '@/components/loading-dialog';
import CurrentResults from '@/components/current-results';
import { FlaskConical, Trash2 } from 'lucide-react';
import QuickAlert from '@/components/quick-alert';

export default function App() {
  const router = useRouter();
  const [files, setFiles] = useState<File[]>([]);
  const [open, setOpen] = useState(false);
  const [qVOpen, setQVOpen] = useState(false);
  const [resetButtonLoading, setResetButtonLoading] = useState(false);
  const [resultsAvailable, setResultsAvailable] = useState(false);
  const [currentResults, setCurrentResults] = useState({
    sigGenuine: false,
    sigConfidence: 0.0,
    writerSame: false,
    writerConfidence: 0.0,
  });
  const [loadingMessage, setLoadingMessage] = useState('Identifying Writer Quirks...');

  useEffect(() => {
    const result = localStorage.getItem('quick_result');
    if (result) {
      setResultsAvailable(true);
      setCurrentResults(JSON.parse(result));
    } else {
      setQVOpen(true);
      return;
    }
    async function fetchData() {
      try {
        const res = await fetch(process.env.NEXT_PUBLIC_API + 'train/status');
        if (!res.ok) {
            router.push('/personalized-verification/train');
        }
      } catch (err) {
        console.error('Fetch error:', err);
      }
    }
    fetchData();
  }, []);
  async function handleVerifier() {
    if (files.length < 1) {
      toast.error('Please upload a test sample.');
      return;
    }
    setOpen(true);
    const formData = new FormData();
    const field = `file`;
    const ext = files[0].name.includes('.')
      ? files[0].name.slice(files[0].name.lastIndexOf('.'))
      : '.png';
    const filename = `test${ext}`;
    formData.append(field, files[0], filename);
    const blob = new Blob([JSON.stringify(currentResults)], { type: 'application/json' });
    formData.append('quick_result', blob, 'quick_result.json');
    try {
      const res = await fetch(
        process.env.NEXT_PUBLIC_API + 'predict/personalized-writer-verification/create',
        {
          method: 'POST',
          body: formData,
        }
      );
      if (!res.ok) throw new Error(`Create Failed. ${res.status} ${res.statusText}`);
      const respose = await res.json();
      const taskId = respose.task_id;
      const taskData = {
        task_id: taskId,
      }
      setLoadingMessage('Analyzing Writer Data...');
      const exp_res = await fetch(
        process.env.NEXT_PUBLIC_API + 'predict/personalized-writer-verification/explain',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(taskData),
        }
      );
      if (!exp_res.ok) throw new Error(`Explain Failed. ${res.status} ${res.statusText}`);
      setLoadingMessage('Creating Explanations...');
      const int_res = await fetch(
        process.env.NEXT_PUBLIC_API + 'predict/personalized-writer-verification/interpret',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(taskData),
        }
      );
      if (!int_res.ok) throw new Error(`Interpret Failed. ${res.status} ${res.statusText}`);
      router.push('/personalized-verification/result/' + taskId);
      setOpen(false);
    } catch (err: any) {
      setOpen(false);
      toast.error('Writer verification failed. Please try again later.');
      console.log(err);
    }
  }

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
    <div className='flex flex-col min-h-screen pt-10'>
      {/* Main content */}
      <main className='flex-grow'>
        <div className='hero h-full'>
          <div className='hero-content'>
            <div className='max-w-xl'>
              <h1 className='text-3xl font-bold text-center mt-10'>
                Personalized Writer Verification
              </h1>
              <p className='py-6 text-md'>
                Your model has been trained with the known samples you provided.
                Now, you can upload a test sample to verify the writer's
                identity.
              </p>
              {resultsAvailable && <CurrentResults data={currentResults} />}
              <h2 className='py-6 text-md font-bold'>
                Upload your test sample here
              </h2>
              <FileUploadComponent
                limit={1}
                value={files}
                onValueChange={setFiles}
              />
              <br />
              <br />
              <div className='text-center w-full justify-center'>
                <button
                  className='btn btn-primary btn-soft btn-lg btn-wide disabled:opacity-50 disabled:'
                  onClick={handleVerifier}
                >
                  <FlaskConical /> Test Sample
                </button>
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
                <LoadingDialog open={open} title={loadingMessage} />
                <QuickAlert open={qVOpen} />
              </div>
            </div>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
}
