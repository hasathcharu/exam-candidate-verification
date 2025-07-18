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
import Link from 'next/link';
import LoadingDialog from '@/components/loading-dialog';
import { useRouter } from 'next/navigation';

export default function App() {
  const [files, setFiles] = useState<File[]>([]);
  const [open, setOpen] = useState(false);
  const [trainingComplete, setTrainingComplete] = useState(false);
  const router = useRouter();
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
      setTrainingComplete(true);
    } catch (err: any) {
      setOpen(false);
      toast.error('Model training failed. Please try again later.');
      console.log(err);
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
                This verification takes a bit longer than the normal
                verification, as it provides reasoning behind model decisions.
                For best results, please upload at least 3 known samples of the
                writer you want to verify. A model will be trained on these
                samples.
              </p>
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
                  <svg
                    xmlns='http://www.w3.org/2000/svg'
                    width='24px'
                    height='24px'
                    viewBox='0 0 24 24'
                    fill='none'
                  >
                    <path
                      d='M14 22V16.9612C14 16.3537 13.7238 15.7791 13.2494 15.3995L11.5 14M11.5 14L13 7.5M11.5 14L10 13M13 7.5L11 7M13 7.5L15.0426 10.7681C15.3345 11.2352 15.8062 11.5612 16.3463 11.6693L18 12M10 13L11 7M10 13L9.40011 16.2994C9.18673 17.473 8.00015 18.2 6.85767 17.8573L4 17M11 7L8.10557 8.44721C7.428 8.786 7 9.47852 7 10.2361V12M14.5 3.5C14.5 4.05228 14.0523 4.5 13.5 4.5C12.9477 4.5 12.5 4.05228 12.5 3.5C12.5 2.94772 12.9477 2.5 13.5 2.5C14.0523 2.5 14.5 2.94772 14.5 3.5Z'
                      strokeWidth='2'
                      stroke='currentColor'
                      strokeLinecap='round'
                      strokeLinejoin='round'
                    />
                  </svg>
                  Train Verifier
                </button>
                <AlertDialog
                  open={open}
                  onOpenChange={(next) => next && setOpen(true)}
                >
                  <AlertDialogContent>
                    {trainingComplete ? (
                      <span>
                        {' '}
                        <>
                          <div className='text-center'>
                            <AlertDialogTitle>
                              Training Complete!
                            </AlertDialogTitle>
                            <br/>
                            <Link href='/advanced-verification'>
                                <button
                                    className='btn btn-primary btn-soft btn-lg btn-wide'
                                    onClick={()=> router.push('/advanced-verification/result')}
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
                            </Link>
                          </div>
                        </>
                      </span>
                    ) : (
                    <LoadingDialog title='Training Model...' />
                    )}
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
