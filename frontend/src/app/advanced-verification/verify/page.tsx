'use client';
import Footer from '@/components/footer';
import { FileUploadComponent } from '@/components/file-upload-component';
import { useState } from 'react';
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { toast } from 'sonner';
import { useRouter } from 'next/navigation';
import LoadingDialog from '@/components/loading-dialog';

export default function App() {
  const router = useRouter();
  const [files, setFiles] = useState<File[]>([]);
  const [open, setOpen] = useState(false);
  const [resetButtonLoading, setResetButtonLoading] = useState(false);
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
    try {
      const res = await fetch(
        process.env.NEXT_PUBLIC_API + 'predict/advanced-writer-verification',
        {
          method: 'POST',
          body: formData,
        }
      );
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const payload = await res.json();
      if (res.status !== 200) {
        throw new Error(payload.message || 'Training failed');
      }
      router.push('/advanced-verification/result');
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
      router.push('/advanced-verification');
      setResetButtonLoading(false);
    } catch (err: any) {
      toast.error('Resetting model failed. Something went wrong.');
      console.log(err);
      setResetButtonLoading(false);
    }
  }
  return (
    <div className='flex flex-col h-screen pt-10'>
      {/* Main content */}
      <main className='flex-grow'>
        <div className='hero h-full'>
          <div className='hero-content'>
            <div className='max-w-xl'>
              <h1 className='text-3xl font-bold text-center'>
                Advanced Writer Verification
              </h1>
              <p className='py-6 text-md'>
                Your model has been trained with the known samples you provided.
                Now, you can upload a test sample to verify the writer's
                identity.
              </p>
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
                  Test Sample
                </button>
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
                <AlertDialog
                  open={open}
                  onOpenChange={(next) => next && setOpen(true)}
                >
                  <AlertDialogContent>
                    <LoadingDialog title='Generating Report...' />
                  </AlertDialogContent>
                </AlertDialog>
              </div>
            </div>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
}
