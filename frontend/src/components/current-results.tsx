import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import NvResults from './nv-results';
import { Trash2 } from 'lucide-react';
import { useRouter } from 'next/navigation';

export default function ({ data }: { data: any }) {
    const router = useRouter();
  return (
    <Accordion type='single' collapsible>
      <AccordionItem value='item-1'>
        <AccordionTrigger className='cursor-pointer text-md font-bold'>
          Current Results
        </AccordionTrigger>
        <AccordionContent className='flex gap-2 flex-col'>
          <NvResults data={data} />
          <div className='flex justify-end'>
            <button
              className='btn btn-ghost'
              onClick={() => {
                localStorage.removeItem('quick_result');
                router.push('/quick-verification');
              }}
            >
              <Trash2 /> Reset
            </button>
          </div>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}
