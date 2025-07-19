import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import NvResults from './nv-results';

export default function ({ data }: { data: any }) {
  return (
    <Accordion type='single' collapsible>
      <AccordionItem value='item-1'>
        <AccordionTrigger className='cursor-pointer text-md font-bold'>
          Current Results
        </AccordionTrigger>
        <AccordionContent className='flex items-center justify-center'>
          <NvResults data={data} />
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}
