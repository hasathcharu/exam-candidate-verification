import ResultCard from '@/components/ui/result-card';

export default function ({data}: {data: any}) {
  return (
    <div className='flex flex-col gap-5 w-xl'>
      <ResultCard
        result={data.sigGenuine}
        confidence={data.sigConfidence}
        type='module1'
      />
      <ResultCard
        result={data.writerSame}
        confidence={data.writerConfidence}
        type='module2'
      />
    </div>
  );
}
