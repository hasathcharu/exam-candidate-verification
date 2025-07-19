'use client';
import Footer from '@/components/footer';
import { FileUploadComponent } from '@/components/file-upload-component';
import { useState } from 'react';
import {
  AlertDialog,
  AlertDialogContent,
} from '@/components/ui/alert-dialog';
import { toast } from 'sonner';
import { useRouter } from 'next/navigation';
import LoadingDialog from '@/components/loading-dialog';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

export default function App() {
  const router = useRouter();
  const [qNormalFile, setQNormalFile] = useState<File[]>([]);
  const [qFastFile, setQFastFile] = useState<File[]>([]);
  const [kNormalFile, setKNormalFile] = useState<File[]>([]);
  const [kFastFile, setKFastFile] = useState<File[]>([]);
  const [questionedFile, setQuestionedFile] = useState<File[]>([]);
  const [knownFile, setKnownFile] = useState<File[]>([]);
  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState('one-s');

  async function handleVerifier() {
    if (tab === 'one-s') {
      if (questionedFile.length < 1 || knownFile.length < 1) {
        toast.error('Please upload both samples.');
        return;
      }
      setOpen(true);
      const formData = new FormData();
      formData.append('file1', questionedFile[0]);
      formData.append('file2', knownFile[0]);
      try {
        const response = await fetch(
          process.env.NEXT_PUBLIC_API + 'predict/automatic-writer-verification',
          {
            method: 'POST',
            body: formData,
          }
        );
        if (!response.ok) {
          throw new Error('Server error');
        }
        const data = await response.json();
        console.log('Verification result:', data);
        const same_writer = data.same_writer;
        const probability = data.probability;
        toast.success('Verification completed!');
      } catch (error) {
        console.error(error);
        toast.error('An error occurred during verification.');
      } finally {
        setOpen(false);
      }
    } else if (tab === 'two-s') {
      if (
        qNormalFile.length < 1 ||
        qFastFile.length < 1 ||
        kNormalFile.length < 1 ||
        kFastFile.length < 1
      ) {
        toast.error('Please upload all four samples.');
        return;
      }
      setOpen(true);
      const formData = new FormData();
      formData.append('file1_normal', qNormalFile[0]);
      formData.append('file1_fast', qFastFile[0]);
      formData.append('file2_normal', kNormalFile[0]);
      formData.append('file2_fast', kFastFile[0]);
      try {
        const response = await fetch(
          process.env.NEXT_PUBLIC_API + 'predict/pairwise-writer-verification',
          {
            method: 'POST',
            body: formData,
          }
        );
        if (!response.ok) {
          throw new Error('Server error');
        }
        const data = await response.json();
        console.log('Two-speed verification result:', data);
        const same_writer = data.same_writer;
        const probability = data.probability;
        toast.success('Verification completed!');
      } catch (error) {
        console.error(error);
        toast.error('An error occurred during verification.');
      } finally {
        setOpen(false);
      }
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
              <h3 className='text-1xl font-bold'>Handwriting</h3>
              <p className='py-6 text-md'>
                Please upload the signature sample you wish to verify. The
                system will determine whether the signature appears genuine or
                forged with the confidence level.
              </p>
              <Tabs value={tab} onValueChange={setTab}>
                <TabsList>
                  <TabsTrigger value='one-s'>Standard Verification</TabsTrigger>
                  <TabsTrigger value='two-s'>
                    Two Speed Verification
                  </TabsTrigger>
                </TabsList>
                <TabsContent value='one-s'>
                  Upload a questioned sample and a known reference. We’ll
                  compare them to see if the handwriting matches.
                  <div className='mt-4 flex gap-4'>
                    <div>
                      <span className='text-sm font-medium'>
                        Questioned Sample
                      </span>
                      <FileUploadComponent
                        limit={1}
                        value={questionedFile}
                        onValueChange={setQuestionedFile}
                      />
                    </div>
                    <div>
                      <span className='text-sm font-medium'>Known Sample</span>
                      <FileUploadComponent
                        limit={1}
                        value={knownFile}
                        onValueChange={setKnownFile}
                      />
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value='two-s'>
                  If you have got normal and fast speed writing from questioned
                  and known samples, you can upload them here to get verified.
                  <div className='mt-4 flex gap-4'>
                    <div>
                      <div className='text-md font-medium h-7'>
                        Questioned Sample
                      </div>
                      <div className='text-sm font-medium'>Normal Speed</div>
                      <FileUploadComponent
                        limit={1}
                        value={qNormalFile}
                        onValueChange={setQNormalFile}
                      />
                    </div>
                    <div>
                      <div className='text-md font-medium h-7'>&nbsp;</div>
                      <div className='text-sm font-medium'>Fast Speed</div>
                      <FileUploadComponent
                        limit={1}
                        value={qFastFile}
                        onValueChange={setQFastFile}
                      />
                    </div>
                  </div>
                  <div className='mt-4 flex gap-4'>
                    <div>
                      <div className='text-md font-medium h-7'>
                        Known Sample
                      </div>
                      <div className='text-sm font-medium'>Normal Speed</div>
                      <FileUploadComponent
                        limit={1}
                        value={kNormalFile}
                        onValueChange={setKNormalFile}
                      />
                    </div>
                    <div>
                      <div className='text-md font-medium h-7'>&nbsp;</div>
                      <div className='text-sm font-medium'>Fast Speed</div>
                      <FileUploadComponent
                        limit={1}
                        value={kFastFile}
                        onValueChange={setKFastFile}
                      />
                    </div>
                  </div>
                </TabsContent>
              </Tabs>

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
                    <LoadingDialog title='Verifying Handwriting...' />
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
