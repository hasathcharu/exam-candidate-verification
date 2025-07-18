import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const Markdown = ({ text }: { text: string }) => {
  return (
    <div className='prose mx-auto max-w-3xl text-black'>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>
        {text.replace(/\\n/g, '\n')}
      </ReactMarkdown>
    </div>
  );
};

export default Markdown;
