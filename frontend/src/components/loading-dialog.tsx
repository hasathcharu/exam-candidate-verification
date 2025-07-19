import { AlertDialogTitle } from './ui/alert-dialog';
import Orb from './orb';
import { AlertDialogDescription } from '@radix-ui/react-alert-dialog';
export default function ({ title }: { title: string }) {
  return (
    <div className='text-center'>
      <AlertDialogDescription>
      </AlertDialogDescription>
      <Orb />
      <AlertDialogTitle>{title}</AlertDialogTitle>
    </div>
  );
}
