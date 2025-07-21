'use client';
import Footer from '@/components/footer';
import { FileUploadComponent } from '@/components/file-upload-component';
import { useEffect, useState } from 'react';
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { toast } from 'sonner';
import Link from 'next/link';
import LoadingDialog from '@/components/loading-dialog';
import { useRouter } from 'next/navigation';
import CurrentResults from '@/components/current-results';
import { BrainCircuit, FlaskConical } from 'lucide-react';
import QuickAlert from '@/components/quick-alert';

export default function App() {
  const [files, setFiles] = useState<File[]>([]);
  const [open, setOpen] = useState(false);
  const [qVOpen, setQVOpen] = useState(false);
  const [trainingComplete, setTrainingComplete] = useState(false);
  const router = useRouter();
  const [resultsAvailable, setResultsAvailable] = useState(false);
  const [currentResults, setCurrentResults] = useState({
    sigGenuine: false,
    sigConfidence: 0.0,
    writerSame: false,
    writerConfidence: 0.0,
  });

  useEffect(() => {
    const result = localStorage.getItem('quick_result');
    if (result) {
      setResultsAvailable(true);
      setCurrentResults(JSON.parse(result));
    } else {
      setQVOpen(true);
    }
  }, []);

  async function handleTrainVerifier() {
    if (files.length < 2) {
      toast.error(
        'Please upload at least 2 known samples to train the verifier.'
      );
      return;
    }
    setOpen(true);
    const formData = new FormData();
    files.forEach((file, idx) => {
      const field = `file${idx + 1}`;
      const ext = file.name.includes('.')
        ? file.name.slice(file.name.lastIndexOf('.'))
        : '.png';
      const filename = `sample${idx + 1}${ext}`;
      formData.append(field, file, filename);
    });
    try {
      const res = await fetch(process.env.NEXT_PUBLIC_API + 'train', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const payload = await res.json();
      if (res.status !== 200) {
        throw new Error(payload.message || 'Training failed');
      }
      setOpen(false);
      setTrainingComplete(true);
    } catch (err: any) {
      setOpen(false);
      toast.error('Model training failed. Please try again later.');
      console.log(err);
    }
  }
  console.log(files);
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
                This verification takes a bit longer than the normal
                verification, as it provides reasoning behind model decisions.
                For best results, please upload at least 3 known samples of the
                writer you want to verify. A model will be trained on these
                samples.
              </p>
              {resultsAvailable && <CurrentResults data={currentResults} />}
              <h2 className='py-6 text-md font-bold'>
                Upload known samples here
              </h2>
              <FileUploadComponent
                limit={10}
                value={files}
                onValueChange={setFiles}
              />
              <br />
              <br />
              <div className='text-center w-full'>
                <button
                  className='btn btn-primary btn-soft btn-lg btn-wide'
                  onClick={handleTrainVerifier}
                >
                  <BrainCircuit /> Train Verifier
                </button>
                <AlertDialog open={trainingComplete}>
                  <AlertDialogContent>
                    <div className='text-center'>
                      <AlertDialogTitle>Training Complete!</AlertDialogTitle>
                      <br />
                      <p className='text-sm text-justify mb-4'>
                        Model is trained successfully. You can now upload a test
                        sample to verify the writer's identity.
                      </p>
                      <Link href='/personalized-verification'>
                        <button className='btn btn-primary btn-soft btn-lg btn-wide'>
                          <FlaskConical /> Test Sample
                        </button>
                      </Link>
                    </div>
                  </AlertDialogContent>
                </AlertDialog>
                <LoadingDialog open={open} title='Training Model...' />
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
