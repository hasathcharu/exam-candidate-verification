export default function Emphasis({
  children,
  type,
}: {
  children: React.ReactNode;
  type: 'green' | 'red' | 'ai';
}) {
  switch (type) {
    case 'green':
      return (
        <span className='bg-gradient-to-br from-lime-400 to-lime-500 py-1 px-2 rounded-sm'>{children}</span>
      );
    case 'red':
      return (
        <span className='bg-gradient-to-br from-rose-400 to-red-400 rounded-sm px-2 py-1'>{children}</span>
      );
    case 'ai':
      return (
        <span className='bg-gradient-to-br from-blue-200 to-cyan-200 py-1 px-2 rounded-sm'>{children}</span>
      );
  }
}
