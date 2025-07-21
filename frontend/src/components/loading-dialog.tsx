import {
  AlertDialogTitle,
  AlertDialogDescription,
  AlertDialog,
  AlertDialogContent,
} from './ui/alert-dialog';
import Orb from './orb';
import { AnimatePresence, motion } from 'framer-motion';
export default function ({ open, title }: { open: boolean; title: string }) {
  return (
    <AlertDialog open={open}>
      <AnimatePresence mode='wait'>
        <motion.div>
          <AlertDialogContent className='focus:outline-none focus:ring-0 focus:border-gray-200'>
            <AlertDialogDescription></AlertDialogDescription>
            <Orb />
            <AlertDialogTitle>
              <AnimatePresence mode='wait'>
                <motion.div
                  key={title}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.3, ease: 'easeInOut' }}
                  className='text-center'
                >
                  {title}
                </motion.div>
              </AnimatePresence>
            </AlertDialogTitle>
          </AlertDialogContent>
        </motion.div>
      </AnimatePresence>
    </AlertDialog>
  );
}
