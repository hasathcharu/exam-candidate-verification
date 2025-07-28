import Link from 'next/link';
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogTitle,
} from './ui/alert-dialog';
import { FlaskConical } from 'lucide-react';

export default function ({ open }: { open: boolean }) {
  return (
    <AlertDialog open={open}>
      <AlertDialogContent className='gap-3'>
        <AlertDialogTitle>Try with Quick Verification</AlertDialogTitle>
        <AlertDialogDescription>
          Personalized verification requires a quick verification to be
          performed first. Therfore, please perform the quick verification
          first.
        </AlertDialogDescription>
        <div className='text-right mt-2'>
          <Link href='/quick-verification'>
            <button className='btn btn-info btn-soft'>
              <FlaskConical /> Quick Verification
            </button>
          </Link>
        </div>
      </AlertDialogContent>
    </AlertDialog>
  );
}
