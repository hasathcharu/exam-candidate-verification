import Link from 'next/link';
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogTitle,
} from './ui/alert-dialog';
import { FlaskConical } from 'lucide-react';

export default function ({ open }: { open: boolean }) {
  return (
    <AlertDialog open={open}>
      <AlertDialogContent>
        <>
          <div className='text-center'>
            <AlertDialogTitle>Try with Quick Verification</AlertDialogTitle>
            <br />
            <p className='text-sm text-justify mb-4'>
              Personalized verification requires a quick verification to be performed first. Therfore, please perform the quick verification first.
            </p>
            <Link href='/quick-verification'>
              <button
                className='btn btn-info btn-soft btn-lg btn-wide'
              >
                <FlaskConical /> Quick Verification
              </button>
            </Link>
          </div>
        </>
      </AlertDialogContent>
    </AlertDialog>
  );
}
