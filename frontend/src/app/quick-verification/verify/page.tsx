'use client';
import Footer from '@/components/footer';
import { FileUploadComponent } from '@/components/file-upload-component';
import { useState } from 'react';
import { AlertDialog, AlertDialogContent } from '@/components/ui/alert-dialog';
import { toast } from 'sonner';
import LoadingDialog from '@/components/loading-dialog';

export default function App() {
  const [files, setFiles] = useState<File[]>([]);
  const [open, setOpen] = useState(false);
  const [label, setLabel] = useState('');
  const [confidence, setConfidence] = useState(0.0);
  async function handleVerifier() {
    if (files.length < 1) {
      toast.error('Please upload a test sample.');
      return;
    }
    setOpen(true);
    const formData = new FormData();
    const field = `sign_file`;
    const ext = files[0].name.includes('.')
      ? files[0].name.slice(files[0].name.lastIndexOf('.'))
      : '.png';
    const filename = `test${ext}`;
    formData.append(field, files[0], filename);
    try {
      const res = await fetch(
        process.env.NEXT_PUBLIC_API + 'predict/signature-verification',
        {
          method: 'POST',
          body: formData,
        }
      );
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const payload = await res.json();
      if (res.status !== 200) {
        throw new Error(payload.message || 'Signature detection failed');
      }
      setLabel(payload.label);
      setConfidence((payload.confidence*100).toFixed(2));
      setOpen(false);
    } catch (err) {
      setOpen(false);
      toast.error('Signature detection failed. Please try again later.');
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
              <h1 className='text-3xl font-bold text-center mb-3'>
                Quick Writer Verification
              </h1>
              <h3 className='text-1xl font-bold'>
                Signature Forgery Detection
              </h3>
              <p className='py-6 text-md'>
                Please upload the signature sample you wish to verify. The
                system will determine whether the signature appears genuine or
                forged with the confidence level.
              </p>
              <h2 className='py-6 text-md font-bold'>
                Upload Questioned Signature Sample
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

                <AlertDialog
                  open={open}
                  onOpenChange={(next) => next && setOpen(true)}
                >
                  <AlertDialogContent>
                    <LoadingDialog title='Detecting signature...' />
                  </AlertDialogContent>
                </AlertDialog>
                {label != "" && confidence} {"%  "} {label}
              </div>
            </div>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
}