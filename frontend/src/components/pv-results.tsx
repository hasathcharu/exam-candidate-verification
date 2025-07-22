import ResultCard from '@/components/ui/result-card-vertical';

export default function ({data}: {data: any}) {
  return (
    <div className='flex gap-3 w-3xl'>
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
      <ResultCard
        result={data.personalizedWriterSame}
        confidence={data.personalizedWriterConfidence}
        type='module3'
      />
    </div>
  );
}
