import { AlertDialogTitle } from './ui/alert-dialog';
import Orb from './orb';
export default function ({title}:{title:string}) {
  return (
    <div className='text-center'>
      <Orb />
      <AlertDialogTitle>{title}</AlertDialogTitle>
    </div>
  );
}
